from imports import *


def normalize(x, return_norm=False, eps=1e-5):
    if type(x) is torch.Tensor:
        norm = x.norm(dim=1).unsqueeze(dim=1) 
    else:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
    unit_vec = x / (norm + eps)
    if return_norm:
        return unit_vec, norm
    else:
        return unit_vec

    
def get_full_edges(pos, batch):
    return pos[batch.edge_index[0, :].t()], pos[batch.edge_index[1, :].t()]
    
    
def get_real_edge_indices(batch):
    l = batch.edge_attr[:, 0]
    edges = batch.edge_index.T
    return edges[l == l.min()]


def get_per_graph_property(batch, property_getter):
    if type(batch) is not Batch:
        batch = Batch.from_data_list([batch])
    return np.array(list(map(property_getter, batch.to_data_list())))


def map_node_indices_to_graph_property(batch, node_indices, property_getter):
    if type(batch) is not Batch:
        batch = Batch.from_data_list([batch])
    return torch.tensor(get_per_graph_property(batch, property_getter), 
                        device=batch.x.device)[batch.batch][torch.tensor(node_indices)]


def map_node_indices_to_node_degrees(real_edges, node_indices):
    node, degrees = np.unique(real_edges[:, 0].detach().cpu().numpy(), return_counts=True)
    return torch.tensor(degrees[node_indices], device=real_edges.device)


def get_counter_clockwise_sorted_angle_vertices(edges, pos):
    if type(pos) is torch.Tensor:
        edges = edges.cpu().detach().numpy()
        pos = pos.cpu().detach().numpy()
    u, v = edges[:, 0], edges[:, 1]
    diff = pos[v] - pos[u]
    diff_normalized = normalize(diff)
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
    real_edges = get_real_edge_indices(batch)
    angles = get_counter_clockwise_sorted_angle_vertices(real_edges, pos)
    u, v1, v2 = angles[:, 0], angles[:, 1], angles[:, 2]
    e1 = normalize(pos[v1] - pos[u])
    e2 = normalize(pos[v2] - pos[u])
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
        edge_counts = map_node_indices_to_graph_property(batch, angles[:,0], lambda g: len(get_real_edge_indices(g)))
        result += (edge_counts,)
    return result[0] if len(result) is 1 else result


def get_resolution_score(theta, node_degrees, node_indices):
    sorted_idx = sorted(np.arange(len(node_indices)), key=lambda e: (node_indices[e], theta[e]))
    u_idx = np.unique(node_indices, return_index=True)[1]
    min_theta = theta[sorted_idx][u_idx]
    min_degree = node_degrees[u_idx]
    return min_theta.mul(min_degree).div(2*np.pi).mean()
    
    
def get_min_angle(theta):
    return theta.min().div(2*np.pi).mul(360)


def rescale_with_minimized_stress(pos, batch, return_scale=False):
    batch = batch.to(pos.device)
    d, w = batch.edge_attr[:, 0], batch.edge_attr[:, 1]
    start, end = get_full_edges(pos, batch)
    diff = end - start
    dist = diff.norm(dim=1)
    scale = (w * d * dist).sum() / (w * dist * dist).sum()
    scaled_pos = pos * scale
    if return_scale:
        return scaled_pos, scale
    return scaled_pos
    