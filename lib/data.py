from .imports import *


def cache(fn):
    def cache_hit(*arg, **kwargs):
        return pickle.load(open(kwargs['cache'], 'rb'))
    def cache_miss(*arg, **kwargs):
        cache = kwargs.pop('cache')
        result = fn(*arg, **kwargs)
        pickle.dump(result, open(cache, 'wb'))
        return result
    def wrapped(*arg, **kwargs):
        return (fn(*arg, **kwargs) if 'cache' not in kwargs 
                else cache_hit(*arg, **kwargs) if os.path.isfile(kwargs['cache']) 
                else cache_miss(*arg, **kwargs))
    return wrapped


def generate_random_index(data_path='data/rome', 
                          index_file='data_index.txt'):
    files = glob.glob(f'{data_path}/*.graphml')
    file_names = list(map(os.path.basename, files))
    random.shuffle(file_names)
    with open(index_file, "w") as fout:
        for f in file_names:
            print(f, file=fout) 
            
            
@cache
def load_G_list(*, data_path, index_file=None, data_slice=slice(None)):
    if index_file is not None:
        all_files = [f'{data_path}/{f}' for f in open(index_file).read().splitlines() if f.rstrip()]
    elif data_path is not None:
        all_files = sorted(glob.glob(f'{data_path}/*.graphml'), 
                           key=lambda x: int(re.search('(?<=grafo)\d+(?=\.)', x).group(0)))
    else:
        return None
    if type(data_slice) is int:
        file_list = [all_files[data_slice]]
    else:
        file_list = all_files[data_slice]
    G_list = []
    for file in tqdm(file_list, desc=f"load G from {data_path}"):
        G = nx.read_graphml(file)
        if nx.is_connected(G):
            mapping = {node: int(node[1:]) for node in G.nodes}
            G = nx.relabel_nodes(G, mapping)
            G_list.append(G)
    return G_list[0] if type(data_slice) is int else G_list


@cache
def generate_data_list(G, *, 
                       sparse=False, 
                       pivot_mode='random', 
                       init_mode='random',
                       edge_index='full_edge_index', 
                       edge_attr='full_edge_attr',
                       pmds_list=None,
                       device='cpu'):
    def generate_pivots(G, apsp, k=None, mode='random'):
        def generate_random_pivots(G, apsp, k):
            return random.sample(list(G.nodes), k)

        def generate_mis_pivots(G, apsp, k):
            diameter = apsp.max()
            k = int(np.floor(np.log2(diameter)))
            V = [[] for _ in range(k)]
            V[0] = list(G.nodes)
            for i in range(1, k):
                d = 2 ** (i - 1) + 1
                V_star = list(V[i-1])
                while len(V_star) > 0:
                    random.shuffle(V_star)
                    p = V_star.pop()
                    V[i].append(p)
                    V_star = [v for v in V_star if apsp[p, v] >= d]
            return V[-1]

        def generate_maxmin_pivots(G, apsp, k):
            nodes = list(G.nodes)
            pivots = [nodes[apsp.max(axis=1).argmax()]]
            for _ in range(k - 1):
                pivots.append(np.argmax([np.min(list(map(lambda p: apsp[i, p], pivots))) for i in nodes]))
            return pivots
        
        methods = {
            'random': generate_random_pivots,
            'mis': generate_mis_pivots,
            'maxmin': generate_maxmin_pivots,
        }
        
        return methods[mode](G, apsp, k)
    
    def generate_maxmin_pivots_new(nodes, apsp, k):
        partial_apsp = apsp[nodes, :][:, nodes]
        pivots = [partial_apsp.max(axis=1).argmax()]
        for _ in range(k - 1):
            pivots.append(partial_apsp[:, pivots].min(axis=1).argmax())
        return np.array(nodes)[pivots]
        
    def get_pivot_groups(G, apsp, pivots):
        # TODO: break ties
        groups = {p: [] for p in pivots}
        for i in G.nodes:
            pivot = pivots[np.argmin([apsp[i, p] for p in pivots])]
            groups[pivot].append(i)
        return groups
    
    def get_pivot_groups_new(nodes, apsp, pivots):
        partial_apsp = apsp[pivots, :][:, nodes]
        groups = np.zeros_like(partial_apsp)
        groups[partial_apsp.argmin(axis=0), np.arange(groups.shape[1])] = 1
#         return {p: (n := np.array(nodes)[g.astype(bool)])[n != p] for p, g in zip(pivots, groups)}
        return {p: np.array(nodes)[g.astype(bool)][np.array(nodes)[g.astype(bool)] != p] for p, g in zip(pivots, groups)}
    
    def get_recursive_pivot_groups(nodes, apsp, k):
        pivots = generate_maxmin_pivots_new(nodes, apsp, k)
        groups = get_pivot_groups_new(nodes, apsp, pivots)
        for p in pivots:
            if len(groups[p]) > k:
                groups[p] = get_recursive_pivot_groups(groups[p], apsp, k)
        return groups
    
    def get_pivot_group_cardinalities(groups):
        cardin_dict = dict()
        def get_cardinalities(groups):
            if type(groups) is not dict:
                cardin_dict.update(zip(groups, np.ones_like(groups)))
                return len(groups) + 1
            cardin = list(map(get_cardinalities, groups.values()))
            cardin_dict.update(zip(groups, cardin))
            return sum(cardin) + 1
        get_cardinalities(groups)
        return np.array(sorted(cardin_dict.items()))[:, 1]
    
    def get_peer_pivot_dense_edge_set(groups):
        edges = set()
        def add_all_edges(groups):
            if type(groups) is dict:
                edges.update([(p, q) for p in groups for q in groups if p != q])
                for p in groups:
                    add_all_edges(groups[p])
        add_all_edges(groups)
        return edges

    def get_same_level_dense_edge_set(groups):
        edges = set()
        def add_all_edges(groups):
            edges.update([(p, q) for p in groups for q in groups if p != q])
            if type(groups) is dict:
                for p in groups:
                    add_all_edges(groups[p])
        add_all_edges(groups)
        return edges

    def get_adjacent_level_dense_edge_set(groups):
        edges = set()
        def add_all_edges(groups):
            if type(groups) is dict:
                descendents = np.concatenate([add_all_edges(groups[p]) for p in groups])
                edges.update([(p, i) for p in groups for i in descendents])
                return np.array(list(groups))
            else:
                return groups
        add_all_edges(groups)
        return edges

    def get_cross_level_dense_edge_set(groups):
        edges = set()
        def add_all_edges(groups):
            if type(groups) is dict:
                descendents = np.concatenate([add_all_edges(groups[p]) for p in groups])
                edges.update([(p, i) for p in groups for i in descendents])
                return np.concatenate((list(groups), descendents))
            else:
                return groups
        add_all_edges(groups)
        return edges

    def get_tree_edge_set(groups):
        edges = set()
        def add_all_edges(groups):
            if type(groups) is dict:
                edges.update([(p, i) for p in groups for i in groups[p]])
                for p in groups:
                    add_all_edges(groups[p])
        add_all_edges(groups)
        return edges

    def get_forward_edge_set(groups):
        edges = set()
        def add_all_edges(groups):
            if type(groups) is dict:
                descendents = {p: add_all_edges(groups[p]) for p in groups}
                edges.update([(p, i) for p in groups for i in descendents[p]])
                return np.concatenate((list(groups), *descendents.values()))
            else:
                return groups
        add_all_edges(groups)
        return edges

    def generate_real_edge_list(G):
        pass
    
    def generate_full_edge_list(G):
        n = G.number_of_nodes()
        return [(i, j)
                for i in range(n) 
                for j in range(n) 
                if i != j]
    
    def generate_sparse_edge_list(G, pivots):
        elist = set()
        for i in range(G.number_of_nodes()):
            for j in pivots:
                if i != j:
                    elist.add((i, j))
                    elist.add((j, i))
        for i, j in list(G.edges):
            elist.add((i, j))
            elist.add((j, i))
        return sorted(list(elist))
    
    def generate_grouped_edge_list(G, pivots, groups):
        elist = set()
        for i in range(G.number_of_nodes()):
            for j in pivots:
                if i != j:
                    elist.add((i, j))
                    elist.add((j, i))
        for i, j in list(G.edges):
            elist.add((i, j))
            elist.add((j, i))
        for p in pivots:
            for i in groups[p]:
                for j in groups[p]:
                    if i != j:
                        elist.add((i, j))
                        elist.add((j, i))
        return sorted(list(elist))
    
    def generate_cluster_edge_list(G, pivots, groups):
        elist = set()
        for i in pivots:
            for j in pivots:
                if i != j:
                    elist.add((i, j))
                    elist.add((j, i))
        for i, j in list(G.edges):
            elist.add((i, j))
            elist.add((j, i))
        for p in pivots:
            for i in groups[p]:
                for j in groups[p]:
                    if i != j:
                        elist.add((i, j))
                        elist.add((j, i))
        return sorted(list(elist))
    
    def get_hierarchical_cluster_edge_set(groups):
        return get_tree_edge_set(groups).union(get_same_level_dense_edge_set(groups))

    def get_recursive_cluster_edge_set(groups):
        return get_forward_edge_set(groups).union(get_same_level_dense_edge_set(groups))

    def get_hierarchical_sparse_edge_set(groups):
        return get_adjacent_level_dense_edge_set(groups).union(get_peer_pivot_dense_edge_set(groups))

    def get_recursive_sparse_edge_set(groups):
        return get_cross_level_dense_edge_set(groups).union(get_peer_pivot_dense_edge_set(groups))
    
    def create_edge_index(*edge_sets, device='cpu'):
        all_edges = reduce(lambda x, y: x.union(y), edge_sets, set())
        reverse_edges = set(map(lambda x: x[::-1], all_edges))
        sorted_symmetric_edges = sorted(list(all_edges.union(reverse_edges)))
        return torch.tensor(sorted_symmetric_edges, dtype=torch.long, device=device).t()
    
    def generate_apsp(G):
        apsp_dict = dict(nx.all_pairs_shortest_path_length(G))
        return np.array([[apsp_dict[j][k] for k in sorted(apsp_dict[j].keys())] for j in sorted(apsp_dict.keys())])
    
    def generate_regular_edge_attr(G, elist, apsp):
        edge_attr = []
        for start, end in elist:
            d = apsp[start, end]
            w = 1 / d**2
            edge_attr.append((d, w))
        return edge_attr
    
    def generate_regular_edge_attr_new(edge_index, apsp, device='cpu'):
        index = edge_index.cpu()
        d = apsp[index[0], index[1]]
        w = 1 / d ** 2
        return torch.tensor(np.stack([d, w], axis=1), dtype=torch.float, device=device)
    
    def generate_pivot_edge_attr_new(edge_index, apsp, cardinalities, device='cpu'):
        index = edge_index.cpu()
        d = apsp[index[0], index[1]]
        s = cardinalities[index[0]]
        w = s / d ** 2
        return torch.tensor(np.stack([d, w], axis=1), dtype=torch.float, device=device)
    
    def generate_symmetric_pivot_edge_attr(edge_index, apsp, cardinalities, device='cpu'):
        index = edge_index.cpu()
        d = apsp[index[0], index[1]]
        s = cardinalities[index[0]] * cardinalities[index[1]]
        w = s / d ** 2
        return torch.tensor(np.stack([d, w], axis=1), dtype=torch.float, device=device)
    
    def generate_pivot_edge_attr(G, sparse_elist, apsp, pivots, groups):
        wdict = {i: {j: 0 if i == j else (1 / apsp[i, j]**2) for j in G.nodes} for i in G.nodes}
        for p in pivots:
            for i in G.nodes:
                if p != i:
                    group = groups[p]
                    d = apsp[p, i]
                    s = np.sum((2 * np.array([apsp[p, j] for j in group])) <= d)
                    w = s / d**2
                    wdict[p][i] = w
        edge_attr = []
        for i, j in sparse_elist:
            d = apsp[i, j]
            w = wdict[i][j]
            edge_attr.append((d, w))
        return edge_attr
    
    def generate_initial_node_attr(G, mode='random'):
        def generate_random_node_attr(G):
            return torch.rand(G.number_of_nodes(), 2)
        
        def generate_pmds_node_attr(G):
            return torch.tensor(pmds_list, dtype=torch.float)
        
        methods = {
            'random': generate_random_node_attr,
            'pmds': generate_pmds_node_attr,
        }
        
        return methods[mode](G)
    
    if type(G) is list:
        return [generate_data_list(g,
                                   sparse=sparse,
                                   pivot_mode=pivot_mode,
                                   init_mode=init_mode,
                                   edge_index=edge_index,
                                   edge_attr=edge_attr,
                                   pmds_list=pmds_list[i],
                                   device=device)
                for i, g in enumerate(tqdm(G, desc='preprocess G'))]
    n = G.number_of_nodes()
    apsp = generate_apsp(G)
    full_elist = generate_full_edge_list(G)
    full_eattr = generate_regular_edge_attr(G, full_elist, apsp)
    data = Data(x=torch.zeros(n, device=device),
                full_edge_index=torch.tensor(full_elist, dtype=torch.long, device=device).t(), 
                full_edge_attr=torch.tensor(full_eattr, dtype=torch.float, device=device))
    if init_mode is not None:
        data.pos = generate_initial_node_attr(G, mode=init_mode).to(device)
    if sparse:
        if sparse == 'sqrt':
            k = np.round(np.sqrt(n))
        elif sparse == 'cbrt':
            k = np.round(n ** (1/3))
        elif sparse == 'log':
            k = np.round(2 * np.log2(n))
        elif callable(sparse):
            k = sparse(G)
        else:
            k = sparse
        pivots = generate_pivots(G, apsp, int(k), mode=pivot_mode)
        groups = get_pivot_groups(G, apsp, pivots)
        
        sparse_elist = generate_sparse_edge_list(G, pivots)
        sparse_eattr = generate_pivot_edge_attr(G, sparse_elist, apsp, pivots, groups)
        sparse_eattr_reg = generate_regular_edge_attr(G, sparse_elist, apsp)
        data.sparse_edge_index = torch.tensor(sparse_elist, dtype=torch.long, device=device).t()
        data.sparse_edge_sparsity = data.sparse_edge_index.shape[1] / data.full_edge_index.shape[1]
        data.sparse_edge_attr = torch.tensor(sparse_eattr, dtype=torch.float, device=device)
        data.sparse_edge_attr_reg = torch.tensor(sparse_eattr_reg, dtype=torch.float, device=device)
        
        grouped_elist = generate_grouped_edge_list(G, pivots, groups)
        grouped_eattr = generate_pivot_edge_attr(G, grouped_elist, apsp, pivots, groups)
        grouped_eattr_reg = generate_regular_edge_attr(G, grouped_elist, apsp)
        data.grouped_edge_index = torch.tensor(grouped_elist, dtype=torch.long, device=device).t()
        data.grouped_edge_sparsity = data.grouped_edge_index.shape[1] / data.full_edge_index.shape[1]
        data.grouped_edge_attr = torch.tensor(grouped_eattr, dtype=torch.float, device=device)
        data.grouped_edge_attr_reg = torch.tensor(grouped_eattr_reg, dtype=torch.float, device=device)
        
        cluster_elist = generate_cluster_edge_list(G, pivots, groups)
        cluster_eattr = generate_pivot_edge_attr(G, cluster_elist, apsp, pivots, groups)
        cluster_eattr_reg = generate_regular_edge_attr(G, cluster_elist, apsp)
        data.cluster_edge_index = torch.tensor(cluster_elist, dtype=torch.long, device=device).t()
        data.cluster_edge_sparsity = data.cluster_edge_index.shape[1] / data.full_edge_index.shape[1]
        data.cluster_edge_attr = torch.tensor(cluster_eattr, dtype=torch.float, device=device)
        data.cluster_edge_attr_reg = torch.tensor(cluster_eattr_reg, dtype=torch.float, device=device)
        
        group_tree = get_recursive_pivot_groups(G.nodes, apsp, k)
        cardinality_map = get_pivot_group_cardinalities(group_tree)
        
        hierarchical_cluster_eset = get_hierarchical_cluster_edge_set(group_tree)
        data.hierarchical_cluster_edge_index = create_edge_index(G.edges, hierarchical_cluster_eset, device=device)
        data.hierarchical_cluster_edge_attr_reg = generate_regular_edge_attr_new(data.hierarchical_cluster_edge_index, apsp, device=device)
        data.hierarchical_cluster_edge_attr_pivot = generate_pivot_edge_attr_new(data.hierarchical_cluster_edge_index, apsp, cardinality_map, device=device)
        data.hierarchical_cluster_edge_attr_sympivot = generate_symmetric_pivot_edge_attr(data.hierarchical_cluster_edge_index, apsp, cardinality_map, device=device)
        data.hierarchical_cluster_edge_sparsity = data.hierarchical_cluster_edge_index.shape[1] / data.full_edge_index.shape[1]
        
        recursive_cluster_eset = get_recursive_cluster_edge_set(group_tree)
        data.recursive_cluster_edge_index = create_edge_index(G.edges, recursive_cluster_eset, device=device)
        data.recursive_cluster_edge_attr_reg = generate_regular_edge_attr_new(data.recursive_cluster_edge_index, apsp, device=device)
        data.recursive_cluster_edge_attr_pivot = generate_pivot_edge_attr_new(data.recursive_cluster_edge_index, apsp, cardinality_map, device=device)
        data.recursive_cluster_edge_attr_sympivot = generate_symmetric_pivot_edge_attr(data.recursive_cluster_edge_index, apsp, cardinality_map, device=device)
        data.recursive_cluster_edge_sparsity = data.recursive_cluster_edge_index.shape[1] / data.full_edge_index.shape[1]
        
        hierarchical_sparse_eset = get_hierarchical_sparse_edge_set(group_tree)
        data.hierarchical_sparse_edge_index = create_edge_index(G.edges, hierarchical_sparse_eset, device=device)
        data.hierarchical_sparse_edge_attr_reg = generate_regular_edge_attr_new(data.hierarchical_sparse_edge_index, apsp, device=device)
        data.hierarchical_sparse_edge_attr_pivot = generate_pivot_edge_attr_new(data.hierarchical_sparse_edge_index, apsp, cardinality_map, device=device)
        data.hierarchical_sparse_edge_attr_sympivot = generate_symmetric_pivot_edge_attr(data.hierarchical_sparse_edge_index, apsp, cardinality_map, device=device)
        data.hierarchical_sparse_edge_sparsity = data.hierarchical_sparse_edge_index.shape[1] / data.full_edge_index.shape[1]
        
        recursive_sparse_eset = get_recursive_sparse_edge_set(group_tree)
        data.recursive_sparse_edge_index = create_edge_index(G.edges, recursive_sparse_eset, device=device)
        data.recursive_sparse_edge_attr_reg = generate_regular_edge_attr_new(data.recursive_sparse_edge_index, apsp, device=device)
        data.recursive_sparse_edge_attr_pivot = generate_pivot_edge_attr_new(data.recursive_sparse_edge_index, apsp, cardinality_map, device=device)
        data.recursive_sparse_edge_attr_sympivot = generate_symmetric_pivot_edge_attr(data.recursive_sparse_edge_index, apsp, cardinality_map, device=device)
        data.recursive_sparse_edge_sparsity = data.recursive_sparse_edge_index.shape[1] / data.full_edge_index.shape[1]
        
    data.edge_index = getattr(data, edge_index)
    data.edge_attr = getattr(data, edge_attr)
    return data