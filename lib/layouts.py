from .imports import *
from .transform import *

def interpolate(layout1, layout2, r):
    return {node: tuple(np.average([layout1[node], layout2[node]], axis=0, weights=(1-r, r))) for node in layout1}

def perturb(layout, r):
    std_pos = np.std(list(layout.values()))
    return {node: tuple(np.array(layout[node]) + np.random.normal(0, std_pos * r, 2)) for node in layout}

def flip_nodes(layout, r):
    k = int(len(layout) * r)
    indices = np.arange(len(layout))
    sample = random.sample(list(indices), k=k)
    shuffled = random.sample(sample, k=k)
    indices[sample] = shuffled
    return dict(zip(layout.keys(), np.array(list(layout.values()))[indices, :].tolist()))

def flip_edges(G, layout, r):
    k = int(G.number_of_edges() * r)
    sample = random.sample(list(G.edges), k=k)
    new_layout = dict(layout)
    for node1, node2 in sample:
        pos1, pos2 = new_layout[node1], new_layout[node2]
        new_layout[node1], new_layout[node2] = pos2, pos1
    return new_layout

def movlsq(layout, n, r):
    v = np.array(list(layout.values()))
    p = np.array(random.sample(list(v), n))

    std_pos = np.std(v)
    distortion = np.random.normal(0, std_pos * r, p.shape)

    q = p + distortion

    p0grid, v0grid = np.meshgrid(p[:,0], v[:,0])
    p1grid, v1grid = np.meshgrid(p[:,1], v[:,1])
    w = 1 / ((p0grid - v0grid) ** 2 + (p1grid - v1grid) ** 2 + 1e-5) 

    p_star = w @ p / np.sum(w, axis=1, keepdims=True)
    q_star = w @ q / np.sum(w, axis=1, keepdims=True)

    p_star_grid = np.repeat(p_star[:,None,:], p.shape[0], axis=1)
    p_grid = np.repeat(p[None, :,:], p_star.shape[0], axis=0)
    p_hat = p_grid - p_star_grid

    q_star_grid = np.repeat(q_star[:,None,:], q.shape[0], axis=1)
    q_grid = np.repeat(q[None, :,:], q_star.shape[0], axis=0)
    q_hat = q_grid - q_star_grid

    spwp = np.einsum('vip,vi,viq->vpq', p_hat, w, p_hat)
    spwq = np.einsum('vip,vi,viq->vpq', p_hat, w, q_hat)

    M = np.linalg.inv(spwp) @ spwq

    distorted_pos = np.einsum('vp,vpq->vq', v - p_star, M) + q_star
    distorted_layout = dict(zip(layout.keys(), distorted_pos.tolist()))
    return distorted_layout

def get_gviz_layout(G):
    layout = nx.nx_agraph.graphviz_layout(G, prog='neato')
    return layout

def get_sfdp_layout(G):
    layout = nx.nx_agraph.graphviz_layout(G, prog="sfdp")
    return layout

def get_fa2_layout(G):
    layout = fa2.forceatlas2_networkx_layout(G)
    return layout

def get_random_normal_layout(G, r=1, proper=None):
    proper_layout = get_proper_layout(G) if proper is None else proper
    mean_pos = np.mean(list(proper_layout.values()))
    std_pos = np.std(list(proper_layout.values()))
    random_layout = {node: tuple(np.random.normal(mean_pos, std_pos, 2)) for node in proper_layout}
    return interpolate(proper_layout, random_layout, r=r)

def get_random_uniform_layout(G, r=1, proper=None):
    proper_layout = get_proper_layout(G) if proper is None else proper
    min_pos = np.min(list(proper_layout.values()), axis=0)
    max_pos = np.max(list(proper_layout.values()), axis=0)
    random_layout = {node: tuple(np.random.uniform(min_pos, max_pos)) for node in proper_layout}
    return interpolate(proper_layout, random_layout, r=r)

def get_phantom_layout(G, r=1, proper=None):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    phantom = nx.generators.random_graphs.gnm_random_graph(n, m)
    phantom_layout = get_proper_layout(phantom)
    proper_layout = get_proper_layout(G) if proper is None else proper
    random_layout = dict(zip(G.nodes, phantom_layout.values()))
    return interpolate(proper_layout, random_layout, r=r)

def get_perturb_layout(G, r=1, proper=None):
    proper_layout = get_proper_layout(G) if proper is None else proper
    return perturb(proper_layout, r=r)

def get_flip_nodes_layout(G, r=1, proper=None):
    proper_layout = get_proper_layout(G) if proper is None else proper
    return flip_nodes(proper_layout, r=r)

def get_flip_edges_layout(G, r=1, proper=None):
    proper_layout = get_proper_layout(G) if proper is None else proper
    return flip_edges(G, proper_layout, r=r)

def get_movlsq_layout(G, r=1, proper=None):
    layout = get_proper_layout(G) if proper is None else proper
    return movlsq(layout, 5, r)

def get_proper_layout(G):
    return get_gviz_layout(G)

def layout_to_pos(layout):
    return torch.tensor(list(layout.values()))

# class LayoutGenerator(ABC):
#     def __init__(self, name, requires=[], normalize=True):
#         self.name = name
#         self.requires = requires
        
#     @abstractmethod
#     def generate(self, G, dependencies):
#         pass
        
#     def __call__(self, G, layouts):
#         dependencies = {r: layouts[r] for r in self.requires}
#         layout = self.generate(G, dependencies)


def get_ground_truth(data, G, prog='neato', scaled=True):
    gt = torch.tensor(list(nx.nx_agraph.graphviz_layout(G, prog=prog).values())).to(data.edge_attr.device)
    if scaled:
        gt = Normalization()(gt, data)
    return gt


def get_pmds_layout(data, G, pmds_bin='hde/pmds', get_raw=False):
    indot = str(nx.nx.nx_pydot.to_pydot(G))
    outdot = subprocess.check_output([pmds_bin], text=True, input=indot)
    G = nx.nx_pydot.from_pydot(pydot.graph_from_dot_data(outdot)[0])
    raw_layout = nx.get_node_attributes(G, 'pos')
    layout = {int(n): tuple(map(float, pos.replace('"', '').split(','))) for n, pos in raw_layout.items()}
    sorted_layout = dict(sorted(layout.items(), key=lambda pair: pair[0]))
    if not get_raw:
        pos = torch.tensor(list(sorted_layout.values()))
        return Normalization()(pos, data)
    return sorted_layout
