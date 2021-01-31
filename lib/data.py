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
                       model_eidx='full_edge_index', 
                       model_eattr='full_edge_attr',
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
            pivots = [random.choice(nodes)]
            for _ in range(k - 1):
                pivots.append(np.argmax([np.min(list(map(lambda p: apsp[i, p], pivots))) for i in nodes]))
            return pivots
        
        methods = {
            'random': generate_random_pivots,
            'mis': generate_mis_pivots,
            'maxmin': generate_maxmin_pivots,
        }
        
        return methods[mode](G, apsp, k)
    
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
    
    def generate_grouped_edge_list(G, pivots):
        pass
    
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
    
    def generate_pivot_edge_attr(G, sparse_elist, apsp, pivots):
        # TODO: break ties
        groups = {p: [] for p in pivots}
        wdict = {i: {j: 1 for j in G.nodes} for i in G.nodes}
        for i in G.nodes:
            pivot = pivots[np.argmin([apsp[i, p] for p in pivots])]
            groups[pivot].append(i)
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
            return torch.tensor(pmds_list)
        
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
                                   model_eidx=model_eidx,
                                   model_eattr=model_eattr,
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
        else:
            k = sparse(G)
        pivots = generate_pivots(G, apsp, int(k), mode=pivot_mode)
        sparse_elist = generate_sparse_edge_list(G, pivots)
        sparse_eattr = generate_pivot_edge_attr(G, sparse_elist, apsp, pivots)
        sparse_eattr_reg = generate_regular_edge_attr(G, sparse_elist, apsp)
        data.sparse_edge_index = torch.tensor(sparse_elist, dtype=torch.long, device=device).t()
        data.sparse_edge_attr = torch.tensor(sparse_eattr, dtype=torch.float, device=device)
        data.sparse_edge_attr_reg = torch.tensor(sparse_eattr_reg, dtype=torch.float, device=device)
    data.model_edge_index = getattr(data, model_eidx)
    data.model_edge_attr = getattr(data, model_eattr)
    return data