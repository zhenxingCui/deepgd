from .._dependencies import *

# import ground
# from .bentley_ottmann import segments_intersections
# Multipoint = ground.base.get_context().multipoint_cls
# Point = ground.base.get_context().point_cls
# Segment = ground.base.get_context().segment_cls


# def bentley_ottmann_xing(G, pos):
#     # convert edge format from tensor to Segment list
#     def to_seg_list(tensor):
#         return list(map(lambda mat: Segment(Point(*mat[0].tolist()), Point(*mat[1].tolist())), tensor))
#     # find edge positions
#     segments = pos[torch.tensor(list(G.edges))]
#     # shrink each edge by small epsilon in order to exclude connected edges
#     offset_segments = segments * (1 - 1e-5) + segments.flip(dims=(1,)) * 1e-5
#     # compute edge crossings
#     intersections = segments_intersections(to_seg_list(offset_segments))
#     return list(set(intersections.keys()))

@dataclass
class Xing(nn.Module):
    eps: float = 1e-5
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
            xing = torch_scatter.scatter(xing, batch_idx)

        return xing if self.reduce is None else self.reduce(xing)

    