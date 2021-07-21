from lib.imports import *
from lib.tools import *
from lib.utils import *
from lib.modules import *
from lib.data import *

import ground
from bentley_ottmann.planar import segments_intersections
Multipoint = ground.base.get_context().multipoint_cls
Point = ground.base.get_context().point_cls
Segment = ground.base.get_context().segment_cls


def get_xing(G, pos):
    # convert edge format from tensor to Segment list
    def to_seg_list(tensor):
        return list(map(lambda mat: Segment(Point(*mat[0].tolist()), Point(*mat[1].tolist())), tensor))
    segments = pos[torch.tensor(list(G.edges))]
    # shrink each edge by small epsilon in order to exclude connected edges
    offset_segments = segments * (1 - 1e-5) + segments.flip(dims=(1,)) * 1e-5
    intersections = segments_intersections(to_seg_list(offset_segments))
    return intersections


def vis_xing(G, pos, intersections):
    G = copy.copy(G)
    edges = list(G.edges)
    attrs = {edges[e]: {'c': 'r', 'w': 2} for e in set(np.array(list(intersections.keys())).flatten())}
    nx.set_edge_attributes(G, attrs)
    colors = [G[u][v]['c'] if 'c' in G[u][v] else 'black' for u,v in edges]
    weights = [G[u][v]['w'] if 'w' in G[u][v] else 1 for u,v in edges]
    nx.draw(G, pos.tolist(), edge_color=colors, width=weights)

def get_n_xing(G, pos):
    return len(get_xing(G, pos))