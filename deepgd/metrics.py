from ._dependencies import *
from .functions import *


EPS = 1e-5


class Stress(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce
        
    def forward(self, node_pos, batch):
        start, end = get_full_edges(node_pos, batch)
        eu = end.sub(start).norm(dim=1)
        d = batch.full_edge_attr[:, 0]
        edge_stress = eu.sub(d).abs().div(d).square()
        index = batch.batch[batch.edge_index[0]]
        graph_stress = torch_scatter.scatter(edge_stress, index)
        return graph_stress if self.reduce is None else self.reduce(graph_stress)
    

class EdgeVar(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce

    def forward(self, node_pos, batch):
        edge_idx = batch.raw_edge_index.T
        start, end = get_raw_edges(node_pos, batch)
        eu = end.sub(start).norm(dim=1)
        edge_var = eu.sub(1).square()
        index = batch.batch[batch.raw_edge_index[0]]
        graph_var = torch_scatter.scatter(edge_var, index, reduce="mean")
        return graph_var if self.reduce is None else self.reduce(graph_var)
    
    
class Occlusion(nn.Module):
    def __init__(self, gamma=1, reduce=torch.mean):
        super().__init__()
        self.gamma = gamma
        self.reduce = reduce
        
    def forward(self, node_pos, batch):
        start, end = get_full_edges(node_pos, batch)
        eu = end.sub(start).norm(dim=1)
        edge_occusion = eu.mul(-self.gamma).exp()
        index = batch.batch[batch.edge_index[0]]
        graph_occusion = torch_scatter.scatter(edge_occusion, index)
        return graph_occusion if self.reduce is None else self.reduce(graph_occusion)
    

class IncidentAngle(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce
    
    def forward(self, node_pos, batch):
        theta, degrees, indices = get_radians(node_pos, batch, 
                                              return_node_degrees=True, 
                                              return_node_indices=True)
        phi = degrees.float().pow(-1).mul(2*np.pi)
        angle_l1 = phi.sub(theta).abs()
        index = batch.batch[indices]
        graph_l1 = torch_scatter.scatter(angle_l1, index)
        return graph_l1 if self.reduce is None else self.reduce(graph_l1)
    

class TSNEScore(nn.Module):
    def __init__(self, sigma=1, reduce=torch.mean):
        super().__init__()
        self.sigma = sigma
        self.reduce = reduce
        
    def forward(self, node_pos, batch):
        p = batch.full_edge_attr[:, 0].div(-2 * self.sigma**2).exp()
        sum_src = torch_scatter.scatter(p, batch.full_edge_index[0])[batch.full_edge_index[0]]
        sum_dst = torch_scatter.scatter(p, batch.full_edge_index[1])[batch.full_edge_index[1]]
        p = (p / sum_src + p / sum_dst) / (2 * batch.n[batch.batch[batch.edge_index[0]]])
        start, end = get_full_edges(node_pos, batch)
        eu = end.sub(start).norm(dim=1)
        index = batch.batch[batch.full_edge_index[0]]
        q = 1 / (1 + eu.square())
        q /= torch_scatter.scatter(q, index)[index]
        edge_kl = (p.log() - q.log()).mul(p)
        graph_kl = torch_scatter.scatter(edge_kl, index)
        return graph_kl if self.reduce is None else self.reduce(graph_kl)


@dataclass
class Xing(nn.Module):
    eps: float = EPS
    scatter: bool = True
    reduce: None = torch.sum
    
    def __post_init__(self):
        super().__init__()
        
    @staticmethod
    def _x(v, u):
        return torch.cross(
            F.pad(v, (0, 1)),
            F.pad(u, (0, 1))
        )[:, -1]
    
    @staticmethod
    def _dot(v, u):
        return (v * u).sum(dim=-1)
    
    def forward(self, pos, batch):
        # get pqrs
        (s1, e1, s2, e2) = batch.edge_pair_index
        p, q = pos[s1], pos[s2]
        r, s = pos[e1] - p, pos[e2] - q

        # shrink by eps
        p += self.eps * r
        q += self.eps * s
        r *= 1 - 2*self.eps
        s *= 1 - 2*self.eps

        # get intersection
        qmp = q - p
        qmpxs = self._x(qmp, s)
        qmpxr = self._x(qmp, r)
        rxs = self._x(r, s)
        rdr = self._dot(r, r)
        t = qmpxs / rxs
        u = qmpxr / rxs
        t0 = self._dot(qmp, r) / rdr
        t1 = t0 + self._dot(s, r) / rdr

        # calculate bool
        zero = torch.zeros_like(rxs)
        parallel = rxs.isclose(zero)
        nonparallel = parallel.logical_not()
        collinear = parallel.logical_and(qmpxr.isclose(zero))

        xing = torch.logical_or(
            collinear.logical_and(
                torch.logical_and(
                    (t0 > 0).logical_or(t1 > 0),
                    (t0 < 1).logical_or(t1 < 1),
                )
            ),
            nonparallel.logical_and(
                torch.logical_and(
                    (0 < t).logical_and(t < 1),
                    (0 < u).logical_and(u < 1),
                )
            )
        ).float()

        if self.scatter:
            batch_idx = batch.batch[s1]
            xing = torch_scatter.scatter(xing, batch_idx, reduce='sum')

        return xing if self.reduce is None else self.reduce(xing)

    
class XingAngle(nn.Module):
    DEFAULT_SCATTER_REDUCE = "sum"
    
    def __init__(self, eps=EPS, scatter=True, scatter_reduce=None, reduce=torch.mean):
        super().__init__()
        self.eps = eps
        self.scatter = scatter
        self.scatter_reduce = scatter_reduce if scatter_reduce is not None else self.DEFAULT_SCATTER_REDUCE
        self.batch_reduce = reduce
    
    @staticmethod
    def _x(v, u):
        return torch.cross(
            F.pad(v, (0, 1)),
            F.pad(u, (0, 1))
        )[:, -1]
    
    @staticmethod
    def _dot(v, u):
        return (v * u).sum(dim=-1)
    
    def forward(self, pos, batch):
        # get pqrs
        (s1, e1, s2, e2) = batch.edge_pair_index
        p, q = pos[s1], pos[s2]
        r, s = pos[e1] - p, pos[e2] - q

        # shrink by eps
        p += self.eps * r
        q += self.eps * s
        r *= 1 - 2*self.eps
        s *= 1 - 2*self.eps

        # get intersection
        qmp = q - p
        qmpxs = self._x(qmp, s)
        qmpxr = self._x(qmp, r)
        rxs = self._x(r, s)
        rdr = self._dot(r, r)
        t = qmpxs / rxs
        u = qmpxr / rxs
        t0 = self._dot(qmp, r) / rdr
        t1 = t0 + self._dot(s, r) / rdr

        # calculate bool
        zero = torch.zeros_like(rxs)
        parallel = rxs.isclose(zero)
        nonparallel = parallel.logical_not()
        collinear = parallel.logical_and(qmpxr.isclose(zero))

        xing = torch.logical_or(
            collinear.logical_and(
                torch.logical_and(
                    (t0 > 0).logical_or(t1 > 0),
                    (t0 < 1).logical_or(t1 < 1),
                )
            ),
            nonparallel.logical_and(
                torch.logical_and(
                    (0 < t).logical_and(t < 1),
                    (0 < u).logical_and(u < 1),
                )
            )
        )
        
        e1 = l2_normalize(r)
        e2 = l2_normalize(s)
        radians = (e1 * e2).sum(dim=1).abs().asin()
        radians[~xing] = 0

        if self.scatter:
            batch_idx = batch.batch[s1]
            radians = torch_scatter.scatter(radians, batch_idx, 
                                            reduce=self.scatter_reduce, 
                                            dim_size=batch.num_graphs)
        
        return radians if self.batch_reduce is None else self.batch_reduce(radians)


@dataclass
class RealXingAngle(nn.Module):
    eps: float = EPS
    scatter: bool = True
    scatter_reduce: str = 'mean'
    batch_reduce: None = torch.mean
    
    def __post_init__(self):
        super().__init__()
    
    @staticmethod
    def _x(v, u):
        return torch.cross(
            F.pad(v, (0, 1)),
            F.pad(u, (0, 1))
        )[:, -1]
    
    @staticmethod
    def _dot(v, u):
        return (v * u).sum(dim=-1)
    
    def forward(self, pos, batch):
        # get pqrs
        (s1, e1, s2, e2) = batch.edge_pair_index
        p, q = pos[s1], pos[s2]
        r, s = pos[e1] - p, pos[e2] - q

        # shrink by eps
        p += self.eps * r
        q += self.eps * s
        r *= 1 - 2*self.eps
        s *= 1 - 2*self.eps

        # get intersection
        qmp = q - p
        qmpxs = self._x(qmp, s)
        qmpxr = self._x(qmp, r)
        rxs = self._x(r, s)
        rdr = self._dot(r, r)
        t = qmpxs / rxs
        u = qmpxr / rxs
        t0 = self._dot(qmp, r) / rdr
        t1 = t0 + self._dot(s, r) / rdr

        # calculate bool
        zero = torch.zeros_like(rxs)
        parallel = rxs.isclose(zero)
        nonparallel = parallel.logical_not()
        collinear = parallel.logical_and(qmpxr.isclose(zero))

        xing = torch.logical_or(
            collinear.logical_and(
                torch.logical_and(
                    (t0 > 0).logical_or(t1 > 0),
                    (t0 < 1).logical_or(t1 < 1),
                )
            ),
            nonparallel.logical_and(
                torch.logical_and(
                    (0 < t).logical_and(t < 1),
                    (0 < u).logical_and(u < 1),
                )
            )
        )
        
        e1 = l2_normalize(r)
        e2 = l2_normalize(s)
        radians = (e1 * e2).sum(dim=1).abs().acos()

        if self.scatter:
            batch_idx = batch.batch[s1]
            noxing = 1 - torch_scatter.scatter(torch.ones_like(radians[xing]), batch_idx[xing], 
                                               reduce='mean', 
                                               dim_size=batch.num_graphs)
            radians = torch_scatter.scatter(radians[xing], batch_idx[xing], 
                                            reduce=self.scatter_reduce, 
                                            dim_size=batch.num_graphs)
            radians += noxing * np.pi / 2

        return radians if self.batch_reduce is None else self.batch_reduce(radians)

    
class Score(nn.Module):
    def __init__(self, weight=None, reduce=torch.mean):
        super().__init__()
        self.weight = weight or {
            'stress': 0.00029434361190166397,
            'xing': 0.0007957665417401572,
            'xangle': 0.0030477412542927345,
            'angle': 0.0006086161066274451,
            'ring': 0.0003978516063712987,
            'edge': 0.5699997020454531,
            'tsne': 0.4248559788336135,
        }
        self.reduce = reduce
        self.metrics = {
            'stress': Stress(reduce=reduce),
            'xing': Xing(reduce=reduce),
            'xangle': XingAngle(reduce=reduce),
            'angle': IncidentAngle(reduce=reduce),
            'ring': Occlusion(reduce=reduce),
            'edge': EdgeVar(reduce=reduce),
            'tsne': TSNEScore(reduce=reduce),
            'gg': GabrielMJD(reduce=reduce),
            'rng': RNGMJD(reduce=reduce),
        }
    
    def forward(self, pos, batch, return_metrics=False):
        metrics = {m: self.metrics[m](pos, batch) for m in self.metrics}
        score = sum([metrics[m] * self.weight[m] for m in self.weight])
        if return_metrics:
            return score, metrics
        return score
    
    
@dataclass
class SPC(nn.Module):
    eps: float = EPS
    reduce: None = torch.mean
        
    def __post_init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)
        spc = (pred - target) / torch.maximum(torch.maximum(pred, target), 
                                              torch.full_like(pred, self.eps))
        if self.reduce:
            return self.reduce(spc)
        return spc

    
def generate_shape_graphs(pos, data, eps=1e-4):
    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy()
    tree = spatial.KDTree(pos)
    try:
        tri = spatial.Delaunay(points=pos)
    except spatial.qhull.QhullError as e:
        data.delaunay_edge_index = data.full_edge_index
        data.gabriel_edge_index = data.full_edge_index
        data.rng_edge_index = data.full_edge_index
        return data
        
    delaunay = np.unique(tri.simplices[:, list(permutations(range(3), 2))].reshape([-1, 2]), axis=0).T

    c = tri.points[delaunay]
    m = c.mean(axis=0)
    d = np.linalg.norm(c[0] - c[1], axis=1)
    r = d / 2

    dm = tree.query(x=m, k=1)[0]
    gabriel = delaunay[:, dm >= r*(1 - eps)]

    p0 = tree.query_ball_point(x=c[0], r=d*(1 - eps))
    p1 = tree.query_ball_point(x=c[1], r=d*(1 - eps))
    p0m = sparse.lil_matrix((delaunay.shape[1], tri.npoints))
    p0m.rows, p0m.data = p0, list(map(np.ones_like, p0))
    p1m = sparse.lil_matrix((delaunay.shape[1], tri.npoints))
    p1m.rows, p1m.data = p1, list(map(np.ones_like, p1))
    rng = delaunay[:, ~(p0m.toarray().astype(bool) & p1m.toarray().astype(bool)).any(axis=1)]
    
    device = data.x.device
    data.delaunay_edge_index = torch.tensor(delaunay).to(device)
    data.gabriel_edge_index = torch.tensor(gabriel).to(device)
    data.rng_edge_index = torch.tensor(rng).to(device)
    
    return data


class GabrielMJD(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce
        
    def forward(self, pos, batch):
        batch.pos = pos
        data_list = batch.to_data_list()
        for data in data_list:
            device = data.x.device
            raw_edge = data.raw_edge_index.cpu().numpy()
            gabriel_edge = generate_shape_graphs(data.pos, data).gabriel_edge_index.cpu().numpy()
            raw_adj = sparse.coo_matrix((np.ones_like(raw_edge[0]), raw_edge), shape=(data.n, data.n)).astype(bool).toarray()
            gabriel_adj = sparse.coo_matrix((np.ones_like(gabriel_edge[0]), gabriel_edge), shape=(data.n, data.n)).astype(bool).toarray()
            data.gabriel_mjd = torch.tensor(1 - np.mean((raw_adj & gabriel_adj).sum(axis=1) / (raw_adj | gabriel_adj).sum(axis=1))).to(device)
        batch = Batch.from_data_list(data_list)
        return batch.gabriel_mjd if self.reduce is None else self.reduce(batch.gabriel_mjd)
            
            
class RNGMJD(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce
        
    def forward(self, pos, batch):
        batch.pos = pos
        data_list = batch.to_data_list()
        for data in data_list:
            device = data.x.device
            raw_edge = data.raw_edge_index.cpu().numpy()
            rng_edge = generate_shape_graphs(data.pos, data).rng_edge_index.cpu().numpy()
            raw_adj = sparse.coo_matrix((np.ones_like(raw_edge[0]), raw_edge), shape=(data.n, data.n)).astype(bool).toarray()
            rng_adj = sparse.coo_matrix((np.ones_like(rng_edge[0]), rng_edge), shape=(data.n, data.n)).astype(bool).toarray()
            data.rng_mjd = torch.tensor(1 - np.mean((raw_adj & rng_adj).sum(axis=1) / (raw_adj | rng_adj).sum(axis=1))).to(device)
        batch = Batch.from_data_list(data_list)
        return batch.rng_mjd if self.reduce is None else self.reduce(batch.rng_mjd)