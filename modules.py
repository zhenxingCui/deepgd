from imports import *
from functions import *

    
class CompositeLossController(nn.Module):
    def __init__(self, criterions, weights=None):
        super().__init__()
        self.weights = np.ones(len(criterions)) if weights is None else weights
        self.criterions = criterions
        
    def __len__(self):
        return len(self.criterions)
        
    def forward(self, *args, return_components=False, return_weights=False, **kwargs):
        losses = 0
        components = []
        for criterion, weight in zip(self.criterions, self.weights):
            loss = criterion(*args, **kwargs)
            losses += loss * weight
            components += [loss]
        result = (losses,)
        if return_components:
            result += (components,)
        if return_weights:
            result += (self.weights,)
        return result[0] if len(result) == 1 else result
        
    def step(self):
        pass
        
        
class GNNLayer(nn.Module):
    def __init__(self, in_vfeat, out_vfeat, in_efeat, edge_net=None, bn=False, act=False, dp=None, aggr='mean'):
        super().__init__()
        self.enet = nn.Linear(in_efeat, in_vfeat * out_vfeat) if edge_net is None else edge_net
        self.conv = gnn.NNConv(in_vfeat, out_vfeat, self.enet, aggr=aggr)
        self.bn = gnn.BatchNorm(out_vfeat) if bn else nn.Identity()
        self.act = nn.LeakyReLU() if act else nn.Identity()
        self.dp = nn.Dropout(dp) if dp is not None else nn.Identity()
        
    def forward(self, v, e, data):
        return self.dp(self.act(self.bn(self.conv(v, data.edge_index, e))))


class GNNBlock(nn.Module):
    def __init__(self, feat_dims, 
                 efeat_hid_dims=[], 
                 efeat_hid_acts=nn.LeakyReLU,
                 bn=False, 
                 act=True, 
                 dp=None, 
                 extra_efeat='skip', 
                 euclidian=False, 
                 direction=False, 
                 residual=False):
        '''
        extra_efeat: {'skip', 'first', 'prev'}
        '''
        super().__init__()
        self.extra_efeat = extra_efeat
        self.euclidian = euclidian
        self.direction = direction
        self.residual = residual
        self.gnn = nn.ModuleList()
        self.n_layers = len(feat_dims) - 1
        
        for idx, (in_feat, out_feat) in enumerate(zip(feat_dims[:-1], feat_dims[1:])):
            direction_dim = feat_dims[idx] if self.extra_efeat == 'prev' else feat_dims[0]
            in_efeat_dim = 2
            if self.extra_efeat != 'first': 
                in_efeat_dim += self.euclidian + self.direction * direction_dim 
            edge_net = nn.Sequential(*chain.from_iterable(
                [nn.Linear(idim, odim),
                 nn.BatchNorm1d(odim),
                 act()]
                for idim, odim, act in zip([in_efeat_dim] + efeat_hid_dims,
                                           efeat_hid_dims + [in_feat * out_feat],
                                           [efeat_hid_acts] * len(efeat_hid_dims) + [nn.Tanh])
            ))
            self.gnn.append(GNNLayer(in_vfeat=in_feat, 
                                     out_vfeat=out_feat, 
                                     in_efeat=in_efeat_dim, 
                                     edge_net=edge_net,
                                     bn=bn, 
                                     act=act, 
                                     dp=dp))
        
    def _get_edge_feat(self, pos, data, euclidian=False, direction=False):
        e = data.edge_attr
        if euclidian or direction:
            start_pos, end_pos = get_full_edges(pos, data)
            u, d = normalize(end_pos - start_pos, return_norm=True)
            if euclidian:
                e = torch.cat([e, u], dim=1)
            if direction:
                e = torch.cat([e, d], dim=1)
        return e
        
    def forward(self, v, data):
        vres = v
        for layer in range(self.n_layers):
            vsrc = v if self.euclidian == 'prev' else vres
            get_extra = not (self.extra_efeat == 'first' and layer != 0)
            e = self._get_edge_feat(vsrc, data, 
                                    euclidian=self.euclidian and get_extra, 
                                    direction=self.direction and get_extra)
            v = self.gnn[layer](v, e, data)
        return v + vres if self.residual else v
        
        
class Model(nn.Module):
    def __init__(self, num_blocks=9):
        super().__init__()
        
        self.in_blocks = nn.ModuleList([
            GNNBlock(feat_dims=[2, 8, 8], bn=True, dp=0.2)
        ])
        self.hid_blocks = nn.ModuleList([
            GNNBlock(feat_dims=[8, 8, 8, 8], 
                     efeat_hid_dims=[16],
                     bn=True, 
                     act=True,
                     dp=0.2, 
                     extra_efeat='skip', 
                     euclidian=True, 
                     direction=True, 
                     residual=True)
            for _ in range(num_blocks)
        ])
        self.out_blocks = nn.ModuleList([
            GNNBlock(feat_dims=[8, 8], bn=True),
            GNNBlock(feat_dims=[8, 2], act=False)
        ])

    def forward(self, data, output_hidden=False, numpy=False, with_initial_pos=False):
        if with_initial_pos:
            v = data.x
        else:
            v = torch.rand_like(data.x) * 2 - 1
                
        v = rescale_with_minimized_stress(v, data)
          
        hidden = []
        for block in chain(self.in_blocks, 
                           self.hid_blocks, 
                           self.out_blocks):
            v = block(v, data)
            if output_hidden:
                hidden.append(v.detach().cpu().numpy() if numpy else v)
        if not output_hidden:
            vout = v.detach().cpu().numpy() if numpy else v
        
        return hidden if output_hidden else vout