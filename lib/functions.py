from .imports import *


def generate_rand_pos(n, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.rand(n, 2).mul(2).sub(1)


def l2_normalize(x, return_norm=False, eps=1e-5):
    if type(x) is torch.Tensor:
        norm = x.norm(dim=1).unsqueeze(dim=1) 
    else:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
    unit_vec = x / (norm + eps)
    if return_norm:
        return unit_vec, norm
    else:
        return unit_vec
    
    
def get_full_edge_index(batch):
    return batch.full_edge_index.T

def get_sparse_edge_index(batch):
    return batch.sparse_edge_index.T
    
def get_real_edge_index(batch):
    l = batch.edge_attr[:, 0]
    return get_full_edge_index(batch)[l == l.min()]


def get_full_edges(node_pos, batch):
    edges = node_pos[get_full_edge_index(batch)]
    return edges[:, 0, :], edges[:, 1, :]

def get_sparse_edges(node_pos, batch):
    edges = node_pos[get_sparse_edge_index(batch)]
    return edges[:, 0, :], edges[:, 1, :]
    
    
def get_real_edges(node_pos, batch):
    edges = node_pos[get_real_edge_index(batch)]
    return edges[:, 0, :], edges[:, 1, :]
    

def get_per_graph_property(batch, property_getter):
    return torch.tensor(list(map(property_getter, batch.to_data_list())), 
                        device=batch.x.device)


def convert_to_padded_batch(x, batch, node_index=..., return_lengths=False):
    lengths = batch.batch[node_index].unique(return_counts=True)[1]
    padded = rnn.pad_sequence(x.split(lengths.tolist()), batch_first=True)
    if return_lengths:
        return padded, lengths
    else:
        return padded
    
    
def map_node_indices_to_graph_property(batch, node_index, property_getter):
    return get_per_graph_property(batch, property_getter)[batch.batch][node_index]


def map_node_indices_to_node_degrees(real_edges, node_indices):
    node, degrees = np.unique(real_edges[:, 0].detach().cpu().numpy(), return_counts=True)
    return torch.tensor(degrees[node_indices], device=real_edges.device)


def get_counter_clockwise_sorted_angle_vertices(edges, pos):
    if type(pos) is torch.Tensor:
        edges = edges.cpu().detach().numpy()
        pos = pos.cpu().detach().numpy()
    u, v = edges[:, 0], edges[:, 1]
    diff = pos[v] - pos[u]
    diff_normalized = l2_normalize(diff)
    # get cosine angle between uv and y-axis
    cos = diff_normalized @ np.array([[1],[0]])
    # get radian between uv and y-axis
    radian = np.arccos(cos) * np.expand_dims(np.sign(diff[:, 1]), axis=1)
    # for each u, sort edges based on the position of v
    sorted_idx = sorted(np.arange(len(edges)), key=lambda e: (u[e], radian[e]))
    sorted_v = v[sorted_idx]
    # get start index for each u
    idx = np.unique(u, return_index=True)[1]
    roll_idx = np.arange(1, len(u) + 1)
    roll_idx[np.roll(idx - 1, -1)] = idx
    rolled_v = sorted_v[roll_idx]
    return np.stack([u, sorted_v, rolled_v]).T[sorted_v != rolled_v]


def get_radians(pos, batch, 
                return_node_degrees=False, 
                return_node_indices=False, 
                return_num_nodes=False, 
                return_num_real_edges=False):
    real_edges = get_real_edge_index(batch)
    angles = get_counter_clockwise_sorted_angle_vertices(real_edges, pos)
    u, v1, v2 = angles[:, 0], angles[:, 1], angles[:, 2]
    e1 = l2_normalize(pos[v1] - pos[u])
    e2 = l2_normalize(pos[v2] - pos[u])
    radians = (e1 * e2).sum(dim=1).acos()
    result = (radians,)
    if return_node_degrees:
        degrees = map_node_indices_to_node_degrees(real_edges, u)
        result += (degrees,)
    if return_node_indices:
        result += (u,)
    if return_num_nodes:
        node_counts = map_node_indices_to_graph_property(batch, angles[:,0], lambda g: g.num_nodes)
        result += (node_counts,)
    if return_num_real_edges:
        edge_counts = map_node_indices_to_graph_property(batch, angles[:,0], lambda g: len(get_real_edge_index(g)))
        result += (edge_counts,)
    return result[0] if len(result) is 1 else result


def get_resolution_score(radians, node_degrees, node_indices):
    sorted_idx = sorted(np.arange(len(node_indices)), key=lambda e: (node_indices[e], radians[e]))
    u_idx = np.unique(node_indices, return_index=True)[1]
    min_theta = radians[sorted_idx][u_idx]
    min_degree = node_degrees[u_idx]
    return min_theta.mul(min_degree).div(2*np.pi).mean()
    
    
def get_min_angle(radians):
    return radians.min().div(2*np.pi).mul(360)


def rescale_with_minimized_stress(pos, batch, return_scale=False):
    batch = batch.to(pos.device)
    d = batch.full_edge_attr[:, 0]
    w = 1/d**2
    start, end = get_full_edges(pos, batch)
    diff = end - start
    dist = diff.norm(dim=1)
    scale = (w * d * dist).sum() / (w * dist * dist).sum()
    scaled_pos = pos * scale
    if return_scale:
        return scaled_pos, scale
    return scaled_pos


def get_ground_truth(data, G, prog='neato', scaled=True):
    gt = torch.tensor(list(nx.nx_agraph.graphviz_layout(G, prog=prog).values())).to(data.edge_attr.device)
    if scaled:
        gt = rescale_with_minimized_stress(gt, data)
    return gt


def get_adj(batch, reverse=False, value=1):
    device = batch.x.device
    adj = torch.zeros(batch.num_nodes, batch.num_nodes).to(device)
    adj[tuple(batch.edge_index)] = 1
    return (1 - adj if reverse else adj) * value

# Hack: will fail for non-complete graphs
def get_complete_adj(batch):
    return get_adj(batch) + np.eye(batch.num_nodes)

def get_shorted_path_adj(batch):
    adj =  get_adj(batch, reverse=True, value=np.inf)
    adj[tuple(batch.edge_index)] = batch.edge_attr[:, 0]
    return adj

def get_edge_length_adj(batch, pos):
    adj =  get_adj(batch, reverse=True, value=np.inf)
    adj[tuple(batch.edge_index)] = (pos[batch.edge_index[0]] - pos[batch.edge_index[1]]).norm(dim=1)
    return adj

def get_num_nodes_adj(batch):
    adj = get_complete_adj(batch)
    adj *= adj.sum(dim=1, keepdim=True)
    return adj

def graph_wise_normalize(batch, mat):
    adj = get_complete_adj(batch)
    sum_mat = adj * mat.sum(dim=0, keepdim=True)
    sum_mat = adj * sum_mat.sum(dim=1, keepdim=True)
    return mat / (sum_mat + 1e-5) 