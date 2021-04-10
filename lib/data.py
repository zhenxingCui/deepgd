from .imports import *
from .tools import *
from .layouts import *


def generate_random_index(data_path='data/rome', 
                          index_file='data_index.txt'):
    files = glob.glob(f'{data_path}/*.graphml')
    file_names = list(map(os.path.basename, files))
    random.shuffle(file_names)
    with open(index_file, "w") as fout:
        for f in file_names:
            print(f, file=fout) 
            
            
@cache()
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


@cache()
def generate_data_list(G, *, 
                       sparse=False, 
                       pivot_mode='random', 
                       init_mode='random',
                       edge_index='full_edge_index', 
                       edge_attr='full_edge_attr',
                       pmds_list=None,
                       gviz_list=None,
                       noisy_layout=False,
                       device='cpu'):
    
    def generate_apsp(G):
        apsp_dict = dict(nx.all_pairs_shortest_path_length(G))
        return np.array([[apsp_dict[j][k] for k in sorted(apsp_dict[j].keys())] for j in sorted(apsp_dict.keys())])
    
    def get_neighborhood_size(apsp):
        return np.cumsum(np.apply_along_axis(lambda x: np.bincount(x, minlength=len(apsp)), 1, apsp), axis=1)
    
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
    
    def generate_maxmin_pivots(nodes, apsp, k):
        partial_apsp = apsp[nodes, :][:, nodes]
        pivots = [partial_apsp.max(axis=1).argmax()]
        for _ in range(k - 1):
            pivots.append(partial_apsp[:, pivots].min(axis=1).argmax())
        return np.array(nodes)[pivots]
        
    def get_pivot_groups(nodes, apsp, pivots):
        partial_apsp = apsp[pivots, :][:, nodes]
        groups = np.zeros_like(partial_apsp)
        groups[partial_apsp.argmin(axis=0), np.arange(groups.shape[1])] = 1
#         return {p: (n := np.array(nodes)[g.astype(bool)])[n != p] for p, g in zip(pivots, groups)}
        return {p: np.array(nodes)[g.astype(bool)][np.array(nodes)[g.astype(bool)] != p] for p, g in zip(pivots, groups)}
    
    def get_recursive_pivot_groups(nodes, apsp, k):
        pivots = generate_maxmin_pivots(nodes, apsp, k)
        groups = get_pivot_groups(nodes, apsp, pivots)
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
        for i in G.nodes:
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
        no_self_loop = {edge for edge in all_edges if edge[0] != edge[1]}
        reverse_edges = set(map(lambda x: x[::-1], no_self_loop))
        sorted_symmetric_edges = sorted(list(no_self_loop.union(reverse_edges)))
        return torch.tensor(sorted_symmetric_edges, dtype=torch.long, device=device).t()
    
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

    def generate_pivot_src_neighborhood_edge_attr(edge_index, apsp, neighborhood, cardinalities, device='cpu'):
        index = edge_index.cpu()
        d = apsp[index[0], index[1]]
        c = cardinalities[index[0]]
        n = neighborhood[index[0], d//2]
        s = np.minimum(c, n)
        w = s / d ** 2
        return torch.tensor(np.stack([d, w], axis=1), dtype=torch.float, device=device)
    
    def generate_pivot_src_group_edge_attr(edge_index, apsp, cardinalities, device='cpu'):
        index = edge_index.cpu()
        d = apsp[index[0], index[1]]
        s = cardinalities[index[0]]
        w = s / d ** 2
        return torch.tensor(np.stack([d, w], axis=1), dtype=torch.float, device=device)
    
    def generate_pivot_srcdst_group_edge_attr(edge_index, apsp, cardinalities, device='cpu'):
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
            return None if pmds_list is None else torch.tensor(pmds_list, dtype=torch.float)
        
        def generate_gviz_node_attr(G):
            return None if gviz_list is None else torch.tensor(gviz_list, dtype=torch.float)
        
        methods = {
            'random': generate_random_node_attr,
            'pmds': generate_pmds_node_attr,
            'gviz': generate_gviz_node_attr,
        }
        
        return methods[mode](G)
    
    def generate_noisy_pos(G, fn, proper):
        r = random.random()
        layout = fn(G, r=r, proper=proper)
        return layout_to_pos(layout), r
    
    if type(G) is list:
        return [generate_data_list(g,
                                   sparse=sparse,
                                   pivot_mode=pivot_mode,
                                   init_mode=init_mode,
                                   edge_index=edge_index,
                                   edge_attr=edge_attr,
                                   pmds_list=pmds_list[i],
                                   gviz_list=gviz_list[i],
                                   noisy_layout=noisy_layout,
                                   device=device)
                for i, g in enumerate(tqdm(G, desc='preprocess G'))]
    n = G.number_of_nodes()
    m = G.number_of_edges()
    apsp = generate_apsp(G)
    neighborhood = get_neighborhood_size(apsp)
    full_elist = generate_full_edge_list(G)
    full_eattr = generate_regular_edge_attr(G, full_elist, apsp)
    data = Data(x=torch.zeros(n, device=device), n=n, m=m,
                raw_edge_index=create_edge_index(G.edges),
                gt_pos = generate_initial_node_attr(G, mode='gviz').to(device),
                full_edge_index=torch.tensor(full_elist, dtype=torch.long, device=device).t(), 
                full_edge_attr=torch.tensor(full_eattr, dtype=torch.float, device=device))
    if init_mode is not None:
        data.pos = generate_initial_node_attr(G, mode=init_mode).to(device)
    if noisy_layout:
        proper = get_proper_layout(G)
        data.random_normal, data.random_normal_r = generate_noisy_pos(G, get_random_normal_layout, proper)
        data.random_uniform, data.random_uniform_r = generate_noisy_pos(G, get_random_uniform_layout, proper)
        data.phantom, data.phantom_r = generate_noisy_pos(G, get_phantom_layout, proper)
        data.perturb, data.perturb_r = generate_noisy_pos(G, get_perturb_layout, proper)
        data.flip_nodes, data.flip_nodes_r = generate_noisy_pos(G, get_flip_nodes_layout, proper)
        data.flip_edges, data.flip_edges_r = generate_noisy_pos(G, get_flip_edges_layout, proper)
        data.movlsq, data.movlsq_r = generate_noisy_pos(G, get_movlsq_layout, proper)
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
        k = int(k)
        
        group_tree = get_recursive_pivot_groups(G.nodes, apsp, 5)
        cardinalities = get_pivot_group_cardinalities(group_tree)
        
        data.pivots = k
        data.hierarchical_pivots = len(cardinalities[cardinalities > 1])
        
        hierarchical_cluster_eset = get_hierarchical_cluster_edge_set(group_tree)
        data.hierarchical_cluster_edge_index = create_edge_index(G.edges, hierarchical_cluster_eset, device=device)
        data.hierarchical_cluster_edge_attr_reg = generate_regular_edge_attr_new(data.hierarchical_cluster_edge_index, apsp, device=device)
        data.hierarchical_cluster_edge_attr_nbhd = generate_pivot_src_neighborhood_edge_attr(data.hierarchical_cluster_edge_index, apsp, neighborhood, cardinalities, device=device)
        data.hierarchical_cluster_edge_attr_pivot = generate_pivot_src_group_edge_attr(data.hierarchical_cluster_edge_index, apsp, cardinalities, device=device)
        data.hierarchical_cluster_edge_attr_sym = generate_pivot_srcdst_group_edge_attr(data.hierarchical_cluster_edge_index, apsp, cardinalities, device=device)
        data.hierarchical_cluster_edge_sparsity = data.hierarchical_cluster_edge_index.shape[1] / data.full_edge_index.shape[1]
        
        recursive_cluster_eset = get_recursive_cluster_edge_set(group_tree)
        data.recursive_cluster_edge_index = create_edge_index(G.edges, recursive_cluster_eset, device=device)
        data.recursive_cluster_edge_attr_reg = generate_regular_edge_attr_new(data.recursive_cluster_edge_index, apsp, device=device)
        data.recursive_cluster_edge_attr_nbhd = generate_pivot_src_neighborhood_edge_attr(data.recursive_cluster_edge_index, apsp, neighborhood, cardinalities, device=device)
        data.recursive_cluster_edge_attr_pivot = generate_pivot_src_group_edge_attr(data.recursive_cluster_edge_index, apsp, cardinalities, device=device)
        data.recursive_cluster_edge_attr_sym = generate_pivot_srcdst_group_edge_attr(data.recursive_cluster_edge_index, apsp, cardinalities, device=device)
        data.recursive_cluster_edge_sparsity = data.recursive_cluster_edge_index.shape[1] / data.full_edge_index.shape[1]
        
        hierarchical_sparse_eset = get_hierarchical_sparse_edge_set(group_tree)
        data.hierarchical_sparse_edge_index = create_edge_index(G.edges, hierarchical_sparse_eset, device=device)
        data.hierarchical_sparse_edge_attr_reg = generate_regular_edge_attr_new(data.hierarchical_sparse_edge_index, apsp, device=device)
        data.hierarchical_sparse_edge_attr_nbhd = generate_pivot_src_neighborhood_edge_attr(data.hierarchical_sparse_edge_index, apsp, neighborhood, cardinalities, device=device)
        data.hierarchical_sparse_edge_attr_pivot = generate_pivot_src_group_edge_attr(data.hierarchical_sparse_edge_index, apsp, cardinalities, device=device)
        data.hierarchical_sparse_edge_attr_sym = generate_pivot_srcdst_group_edge_attr(data.hierarchical_sparse_edge_index, apsp, cardinalities, device=device)
        data.hierarchical_sparse_edge_sparsity = data.hierarchical_sparse_edge_index.shape[1] / data.full_edge_index.shape[1]
        
        recursive_sparse_eset = get_recursive_sparse_edge_set(group_tree)
        data.recursive_sparse_edge_index = create_edge_index(G.edges, recursive_sparse_eset, device=device)
        data.recursive_sparse_edge_attr_reg = generate_regular_edge_attr_new(data.recursive_sparse_edge_index, apsp, device=device)
        data.recursive_sparse_edge_attr_nbhd = generate_pivot_src_neighborhood_edge_attr(data.recursive_sparse_edge_index, apsp, neighborhood, cardinalities, device=device)
        data.recursive_sparse_edge_attr_pivot = generate_pivot_src_group_edge_attr(data.recursive_sparse_edge_index, apsp, cardinalities, device=device)
        data.recursive_sparse_edge_attr_sym = generate_pivot_srcdst_group_edge_attr(data.recursive_sparse_edge_index, apsp, cardinalities, device=device)
        data.recursive_sparse_edge_sparsity = data.recursive_sparse_edge_index.shape[1] / data.full_edge_index.shape[1]
        
        rc_x = create_edge_index(recursive_cluster_eset).shape[1]
        rc_k = int(np.round((2 * n - 1 - np.sqrt((2 * n - 1) ** 2 - 4 * rc_x)) / 2))
        rc_k_pivots = generate_maxmin_pivots(G.nodes, apsp, rc_k)
        rc_k_groups = get_pivot_groups(G, apsp, rc_k_pivots)
        cardinalities_rc_k = get_pivot_group_cardinalities(rc_k_groups)
        data.pivots_rc = rc_k
        
        sparse_eset_rc_k = get_recursive_sparse_edge_set(rc_k_groups)
        data.sparse_edge_index_rc_k = create_edge_index(G.edges, sparse_eset_rc_k, device=device)
        data.sparse_edge_attr_reg_rc_k = generate_regular_edge_attr_new(data.sparse_edge_index_rc_k, apsp, device=device)
        data.sparse_edge_attr_nbhd_rc_k = generate_pivot_src_neighborhood_edge_attr(data.sparse_edge_index_rc_k, apsp, neighborhood, cardinalities_rc_k, device=device)
        data.sparse_edge_attr_pivot_rc_k = generate_pivot_src_group_edge_attr(data.sparse_edge_index_rc_k, apsp, cardinalities_rc_k, device=device)
        data.sparse_edge_attr_sym_rc_k = generate_pivot_srcdst_group_edge_attr(data.sparse_edge_index_rc_k, apsp, cardinalities_rc_k, device=device)
        data.sparse_edge_sparsity_rc_k = data.sparse_edge_index_rc_k.shape[1] / data.full_edge_index.shape[1]
        
        hs_x = create_edge_index(hierarchical_sparse_eset).shape[1]
        hs_k = int(np.round((2 * n - 1 - np.sqrt((2 * n - 1) ** 2 - 4 * hs_x)) / 2))
        hs_k_pivots = generate_maxmin_pivots(G.nodes, apsp, hs_k)
        hs_k_groups = get_pivot_groups(G, apsp, hs_k_pivots)
        cardinalities_hs_k = get_pivot_group_cardinalities(hs_k_groups)
        data.pivots_hs = hs_k
        
        sparse_eset_hs_k = get_recursive_sparse_edge_set(hs_k_groups)
        data.sparse_edge_index_hs_k = create_edge_index(G.edges, sparse_eset_hs_k, device=device)
        data.sparse_edge_attr_reg_hs_k = generate_regular_edge_attr_new(data.sparse_edge_index_hs_k, apsp, device=device)
        data.sparse_edge_attr_nbhd_hs_k = generate_pivot_src_neighborhood_edge_attr(data.sparse_edge_index_hs_k, apsp, neighborhood, cardinalities_hs_k, device=device)
        data.sparse_edge_attr_pivot_hs_k = generate_pivot_src_group_edge_attr(data.sparse_edge_index_hs_k, apsp, cardinalities_hs_k, device=device)
        data.sparse_edge_attr_sym_hs_k = generate_pivot_srcdst_group_edge_attr(data.sparse_edge_index_hs_k, apsp, cardinalities_hs_k, device=device)
        data.sparse_edge_sparsity_hs_k = data.sparse_edge_index_hs_k.shape[1] / data.full_edge_index.shape[1]
        
        pivots = generate_maxmin_pivots(G.nodes, apsp, k)
        groups = get_pivot_groups(G, apsp, pivots)
        cardinalities = get_pivot_group_cardinalities(groups)
        
        sparse_eset = get_recursive_sparse_edge_set(groups)
        data.sparse_edge_index = create_edge_index(G.edges, sparse_eset, device=device)
        data.sparse_edge_attr_reg = generate_regular_edge_attr_new(data.sparse_edge_index, apsp, device=device)
        data.sparse_edge_attr_nbhd = generate_pivot_src_neighborhood_edge_attr(data.sparse_edge_index, apsp, neighborhood, cardinalities, device=device)
        data.sparse_edge_attr_pivot = generate_pivot_src_group_edge_attr(data.sparse_edge_index, apsp, cardinalities, device=device)
        data.sparse_edge_attr_sym = generate_pivot_srcdst_group_edge_attr(data.sparse_edge_index, apsp, cardinalities, device=device)
        data.sparse_edge_sparsity = data.sparse_edge_index.shape[1] / data.full_edge_index.shape[1]
        cluster_eset = get_recursive_cluster_edge_set(groups)
        data.cluster_edge_index = create_edge_index(G.edges, cluster_eset, device=device)
        data.cluster_edge_attr_reg = generate_regular_edge_attr_new(data.cluster_edge_index, apsp, device=device)
        data.cluster_edge_attr_nbhd = generate_pivot_src_neighborhood_edge_attr(data.cluster_edge_index, apsp, neighborhood, cardinalities, device=device)
        data.cluster_edge_attr_pivot = generate_pivot_src_group_edge_attr(data.cluster_edge_index, apsp, cardinalities, device=device)
        data.cluster_edge_attr_sym = generate_pivot_srcdst_group_edge_attr(data.cluster_edge_index, apsp, cardinalities, device=device)
        data.cluster_edge_sparsity = data.cluster_edge_index.shape[1] / data.full_edge_index.shape[1]
        
#         grouped_elist = generate_grouped_edge_list(G, pivots, groups)
#         grouped_eattr = generate_pivot_edge_attr(G, grouped_elist, apsp, pivots, groups)
#         grouped_eattr_reg = generate_regular_edge_attr(G, grouped_elist, apsp)
#         data.grouped_edge_index = torch.tensor(grouped_elist, dtype=torch.long, device=device).t()
#         data.grouped_edge_sparsity = data.grouped_edge_index.shape[1] / data.full_edge_index.shape[1]
#         data.grouped_edge_attr = torch.tensor(grouped_eattr, dtype=torch.float, device=device)
#         data.grouped_edge_attr_reg = torch.tensor(grouped_eattr_reg, dtype=torch.float, device=device)
        
    data.edge_index = data[edge_index]
    data.edge_attr = data[edge_attr]
    return data


def prepare_discriminator_data(data, pos=None, interpolate=0, complete_graph=False):
    dis_data = copy.copy(data)
    if complete_graph:
        dis_data.edge_index = dis_data.full_edge_index
        dis_data.edge_attr = dis_data.full_edge_attr
    else:
        dis_data.edge_index = dis_data.raw_edge_index
    if pos is None:
        dis_data.pos = rescale_with_minimized_stress(dis_data.gt_pos)
    else:
        dis_data.pos = interpolate * rescale_with_minimized_stress(dis_data.gt_pos) + (1 - interpolate) * rescale_with_minimized_stress(pos)
    return dis_data


class LazyDeviceMappingDataLoader:
    def __init__(self, data_list, batch_size, shuffle, device):
        self.loader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
        self.device = device
    
    def __iter__(self):
        for batch in self.loader:
            yield batch.to(self.device)
            
    def __len__(self):
        return len(self.loader)
        
        
class LargeGraphLoader:
    def __init__(self, path, batch_size, slice=slice(None), shuffle=False, device='cpu', edge_index=None, edge_attr=None):
        self.file_list = glob.glob(f'{path}/*.pickle')[slice]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.file_list)
        for i in range(0, len(self.file_list), self.batch_size):
            files = self.file_list[i:min(len(self.file_list), i + self.batch_size)]
            data_list = [pickle.load(open(f, 'rb')) for f in files]
            batch = Batch.from_data_list(data_list)
            if self.edge_index is not None:
                batch.edge_index = batch[self.edge_index]
            if self.edge_attr is not None:
                batch.edge_attr = batch[self.edge_attr]
            yield batch.to(self.device)
        
    def __len__(self):
        return len(self.file_list) // self.batch_size