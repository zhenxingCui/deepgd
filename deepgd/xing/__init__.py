from .._dependencies import *

import ground
from .bentley_ottmann import segments_intersections
Multipoint = ground.base.get_context().multipoint_cls
Point = ground.base.get_context().point_cls
Segment = ground.base.get_context().segment_cls


def bentley_ottmann_xing(G, pos):
    # convert edge format from tensor to Segment list
    def to_seg_list(tensor):
        return list(map(lambda mat: Segment(Point(*mat[0].tolist()), Point(*mat[1].tolist())), tensor))
    # find edge positions
    segments = pos[torch.tensor(list(G.edges))]
    # shrink each edge by small epsilon in order to exclude connected edges
    offset_segments = segments * (1 - 1e-5) + segments.flip(dims=(1,)) * 1e-5
    # compute edge crossings
    intersections = segments_intersections(to_seg_list(offset_segments))
    return list(set(intersections.keys()))


def get_num_xing(pos, batch, eps=1e-5, return_xing=False):
    def x(v, u):
        return torch.cross(
            F.pad(v, (0, 1)),
            F.pad(u, (0, 1))
        )[:, -1]

    def dot(v, u):
        return (v * u).sum(dim=-1)

    # get pqrs
    (s1, e1, s2, e2) = batch.edge_pair_index
    p, q = pos[s1], pos[s2]
    r, s = pos[e1] - p, pos[e2] - q

    # shrink by eps
    p += eps * r
    q += eps * s
    r *= 1 - 2*eps
    s *= 1 - 2*eps

    # get intersection
    qmp = q - p
    qmpxs = x(qmp, s)
    qmpxr = x(qmp, r)
    rxs = x(r, s)
    rdr = dot(r, r)
    t = qmpxs / rxs
    u = qmpxr / rxs
    t0 = dot(qmp, r) / rdr
    t1 = t0 + dot(s, r) / rdr

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

    batch_idx = batch.batch[s1]
    n_xing = torch_scatter.scatter(xing.float(), batch_idx)
    
    return xing if return_xing else n_xing

    