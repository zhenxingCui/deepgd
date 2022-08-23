from ._dependencies import *

from .functions import *
from .transform import *


class SoftAdaptController:
    def __init__(self, criteria, *, initw=None, gamma=None, warmup=2, exploit_rate=0, tau=0.95, beta=1):
        assert warmup >= 2
        assert beta >= 0
        if gamma is None:
            gamma = np.ones(len(criteria))
        self.set_gamma(gamma)
        self.initw = initw
        self.criteria = criteria
        self.warmup = warmup
        self.exploit_rate = exploit_rate
        self.tau = tau
        self.beta = beta
        self.history = []
        self.policy = None
        self.last_policy = None
        self.weights = None
        self.last_weights = None
        self.step()

    def __len__(self):
        return len(self.history)

    @staticmethod
    def _exp_smooth(array, tau):
        array = np.array(array)
        for i in range(1, len(array)):
            array[i] = array[i-1] * tau + array[i] * (1 - tau)
        return array

    @staticmethod
    def _soft_adapt(losses, gamma, beta, eps=1e-5):
        roc = np.diff(losses, axis=0) / (losses[:-1] + eps)
        normalized_roc = roc / (np.abs(roc).sum(axis=1)[:, None] + eps)
        weight = np.array(gamma)[None, :] * np.exp(beta * normalized_roc) / (losses[1:] + eps)
        return weight / (weight.sum(axis=1)[:, None] + eps)

    def _update_policy(self):
        self.last_policy = self.policy
        self.policy = -1 if random.random() < self.exploit_rate else 1

    def step(self, loss_components=None):
        if loss_components is not None:
            self.history.append(loss_components)
        if len(self) >= self.warmup:
            smoothed = self._exp_smooth(self.history, tau=self.tau)
            self._update_policy()
            
            self.last_weights = self.weights
            self.weights = self._soft_adapt(smoothed, gamma=self.gamma, beta=self.policy*self.beta)[-1]
            self.criteria.update_weights(self.weights)
        else:
            self.last_weights = self.weights
            self.weights = np.zeros(self.n_criteria)
            self.weights[0] = 1
            self.criteria.update_weights(self.weights)

    @property
    def n_criteria(self):
        return len(self.criteria)

    def set_gamma(self, gamma):
        self.gamma = np.array(gamma) / np.sum(gamma)

    def get_weights(self):
        return self.weights

    def get_last_weights(self):
        return self.last_weights

    def get_policy(self):
        return self.policy

    def get_last_policy(self):
        return self.last_policy
    

class FixedWeightController:
    def __init__(self, criteria, *, gamma=None, **kwargs):
        self.criteria = criteria
        self.weights = gamma
        self.last_weights = gamma
        self.step()
        
    def __len__(self):
        pass
    
    def step(self, loss_components=None):
        self.criteria.update_weights(self.weights)
        
    @property
    def n_criteria(self):
        return len(self.criteria)
    
    def set_gamma(self, gamma):
        pass
    
    def get_weights(self):
        return self.weights
    
    def get_last_weights(self):
        return self.last_weights
    
    def get_policy(self):
        pass
    
    def get_last_policy(self):
        pass
    
    
class CompositeLoss(nn.Module):
    def __init__(self, criteria, weights=None):
        super().__init__()
        self.weights = np.ones(len(criteria)) if weights is None else np.array(weights)
        self.criteria = criteria
        
    def __len__(self):
        return len(self.criteria)
        
    def update_weights(self, weights):
        self.weights = np.array(weights)
        
    def forward(self, *args, return_components=False, **kwargs):
        losses = []
        components = []
        for criterion, weight in zip(self.criteria, self.weights):
            loss = criterion(*args, **kwargs)
            losses += [loss.mul(weight)]
            components += [loss]
        result = (sum(losses),)
        if return_components:
            result += (components,)
        return result[0] if len(result) == 1 else result
    
    
class AdaptiveWeightCompositeLoss(nn.Module):
    def __init__(self, criteria, importance=None):
        super().__init__()
        self.importance = np.ones(len(criteria)) if importance is None else np.array(importance)
        self.importance = self.importance / sum(self.importance)
        self.criteria = criteria
        
    def __len__(self):
        return len(self.criteria)
        
    def forward(self, *args, return_components=False, **kwargs):
        losses = []
        components = []
        for criterion, imp in zip(self.criteria, self.importance):
            loss = criterion(*args, **kwargs)
            losses.append(loss * imp / loss.item())
            components.append(loss)
        result = sum(losses)
        if return_components:
            return (result, components)
        return result
    
    
class AdaptiveWeightSquareError(nn.Module):
    def __init__(self, importance=None, normalize=True):
        super().__init__()
        self.importance = torch.tensor(1) if importance is None else torch.tensor(importance)
        self.normalize = normalize
        self.mse = nn.MSELoss(reduction='none')
        
    def __len__(self):
        return len(self.importance)
        
    def forward(self, pred, gt):
        error = self.mse(pred, gt)
        mean_err = error.mean(dim=0)
        weight = self.importance.to(pred.device) / mean_err.detach()
        if self.normalize:
            weight /= weight.sum()
        return (mean_err * weight).sum()
    
    
class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False, act=False, dp=None, aggr='mean'):
        super().__init__()
        self.conv = gnn.GCNConv(in_channels, out_channels, aggr=aggr)
        self.bn = gnn.BatchNorm(out_channels) if bn else nn.Identity()
        self.act = nn.LeakyReLU() if act else nn.Identity()
        self.dp = nn.Dropout(dp) if dp is not None else nn.Identity()
        
    def forward(self, x, data):
        x = self.conv(x, data.edge_index)
        x = self.bn(x)
        x = self.act(x)
        x = self.dp(x)
        return x
      
        
class GNNLayer(nn.Module):
    NORM = gnn.GraphNorm
    ACT = nn.GELU
    
    def __init__(self,
                 nfeat_dims,
                 efeat_dim,
                 aggr,
                 edge_net=None, 
                 dense=False,
                 bn=True, 
                 act=True, 
                 dp=None,
                 root_weight=True,
                 skip=True):
        super().__init__()
        try:
            in_dim = nfeat_dims[0]
            out_dim = nfeat_dims[1]
        except:
            in_dim = nfeat_dims
            out_dim = nfeat_dims
        self.enet = nn.Linear(efeat_dim, in_dim * out_dim) if edge_net is None and efeat_dim > 0 else edge_net
        self.conv = gnn.NNConv(in_dim, out_dim, nn=self.enet, aggr=aggr, root_weight=root_weight)
        self.dense = nn.Linear(out_dim, out_dim) if dense else nn.Identity()
        self.bn = self.NORM(out_dim) if bn else nn.Identity()
        self.act = self.ACT() if act else nn.Identity()
        self.dp = dp and nn.Dropout(dp) or nn.Identity()
        self.skip = skip
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        
    def forward(self, v, e, data):
        v_ = v
        v = self.conv(v, data.edge_index, e)
        v = self.dense(v)
        v = self.bn(v)
        v = self.act(v)
        v = self.dp(v)
        return v + self.proj(v_) if self.skip else v
    

class SeparableNNConvLayer(nn.Module):
    
    class ENetWrapper(nn.Module):
        def __init__(self, enet):
            super().__init__()
            self.enet = enet
            
        def forward(self, x):
            return torch.diag_embed(self.enet(x)).flatten(start_dim=1)
            
    def __init__(self,
                 nfeat_dims,
                 efeat_dim,
                 aggr,
                 expansion=1,
                 edge_net=None, 
                 bn=True, 
                 act=nn.LeakyReLU(), 
                 dp=0,
                 root_weight=True,
                 skip=True):
        super().__init__()
        in_dim, hid_dim, out_dim = self._get_dims(nfeat_dims, expansion)
        raw_enet = nn.Linear(efeat_dim, hid_dim) if edge_net is None and efeat_dim > 0 else edge_net
        
        self.theta = nn.Parameter(torch.zeros(hid_dim)[None, :]) if root_weight else 0
        self.enet = self.ENetWrapper(raw_enet)
        self.act = act
        
        self.ipconv = nn.Linear(in_dim, hid_dim)
        self.ibn = gnn.BatchNorm(hid_dim) if bn else nn.Identity()
        
        self.dconv = gnn.NNConv(hid_dim, hid_dim, nn=self.enet, aggr=aggr, root_weight=False)
        self.hbn = gnn.BatchNorm(hid_dim) if bn else nn.Identity()
        
        self.opconv = nn.Linear(hid_dim, out_dim)
        self.obn = gnn.BatchNorm(out_dim) if bn else nn.Identity()
        
        self.proj = None if not skip else nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self.dp = dp and nn.Dropout(dp) or nn.Identity()

    def _get_dims(self, n_dims, expansion):
        try:
            i_dim = n_dims[0]
            o_dim = n_dims[1]
        except:
            i_dim = n_dims
            o_dim = n_dims
        h_dim = o_dim * expansion
        return i_dim, h_dim, o_dim
        
    def forward(self, v, e, data):
        v_ = v
        
        v = self.ipconv(v)
        v = self.ibn(v)
        v = self.act(v)
        
        v = v * self.theta + self.dconv(v, data.edge_index, e)
        v = self.hbn(v)
        v = self.act(v)
        
        v = self.opconv(v)
        v = self.obn(v)
        
        if self.proj is not None:
            v += self.proj(v_)
            
        return self.dp(v)


class GNNBlock(nn.Module):
    def __init__(self, 
                 feat_dims, 
                 efeat_hid_dims=[], 
                 efeat_hid_act=nn.LeakyReLU,
                 efeat_out_act=nn.Tanh,
                 bn=False,
                 act=True,
                 dp=None,
                 aggr='mean',
                 root_weight=True,
                 static_efeats=1,
                 dynamic_efeats='skip',
                 rich_efeats=False,
                 euclidian=False,
                 direction=False,
                 n_weights=0,
                 residual=False):
        '''
        dynamic_efeats: {
            skip: block input to each layer, 
            first: block input to first layer, 
            prev: previous layer output to next layer, 
            orig: original node feature to each layer
        }
        '''
        super().__init__()
        self.static_efeats = static_efeats
        self.dynamic_efeats = dynamic_efeats
        self.rich_efeats = rich_efeats
        self.euclidian = euclidian
        self.direction = direction
        self.n_weights = n_weights
        self.residual = residual
        self.gnn = nn.ModuleList()
        self.n_layers = len(feat_dims) - 1

        for idx, (in_feat, out_feat) in enumerate(zip(feat_dims[:-1], feat_dims[1:])):
            direction_dim = (feat_dims[idx] if self.dynamic_efeats == 'prev'
                             else 2 if self.dynamic_efeats == 'orig'
                             else feat_dims[0])
            in_efeat_dim = self.static_efeats
            if self.dynamic_efeats != 'first': 
                in_efeat_dim += self.euclidian + self.direction * direction_dim + self.n_weights + 3 * self.rich_efeats
            edge_net = nn.Sequential(*chain.from_iterable(
                [nn.Linear(idim, odim),
                 nn.BatchNorm1d(odim),
                 act()]
                for idim, odim, act in zip([in_efeat_dim] + efeat_hid_dims,
                                           efeat_hid_dims + [in_feat * out_feat],
                                           [efeat_hid_act] * len(efeat_hid_dims) + [efeat_out_act])
            ))
            self.gnn.append(GNNLayer(nfeat_dims=(in_feat, out_feat), 
                                     efeat_dim=in_efeat_dim, 
                                     edge_net=edge_net,
                                     bn=bn, 
                                     act=act, 
                                     dp=dp,
                                     aggr=aggr,
                                     root_weight=root_weight,
                                     skip=False))
        
    def _get_edge_feat(self, pos, data, rich_efeats=False, euclidian=False, direction=False, weights=None):
        e = data.edge_attr[:, :self.static_efeats]
        if euclidian or direction:
            start_pos, end_pos = get_edges(pos, data)
            v, u = l2_normalize(end_pos - start_pos, return_norm=True)
            if euclidian:
                e = torch.cat([e, u], dim=1)
            if direction:
                e = torch.cat([e, v], dim=1)
            if rich_efeats:
                d = e[:, :1]
                d2 = d ** 2
#                 d_inv = 1 / d
#                 d2_inv = 1 / d2
                u2 = u ** 2
                ud = u * d
#                 u_inv = 1 / u
#                 u2_inv = 1 / u2
                e = torch.cat([e, d2, u2, ud], dim=1)
        if weights is not None:
            w = weights.repeat(len(e), 1)
            e = torch.cat([e, w], dim=1)
        return e
    
    def _get_dynamic_edge_feat(self, pos, data, rich_efeats=False, euclidian=False, direction=False, weights=None):
        if euclidian or direction:
            start_pos, end_pos = get_edges(pos, data)
            d, u = l2_normalize(end_pos - start_pos, return_norm=True)
            if euclidian and direction:
                e = torch.cat([u, d], dim=1)
            else:
                if euclidian:
                    e = u
                if direction:
                    e = d
        if weights is not None:
            w = weights.repeat(len(e), 1)
            e = torch.cat([e, w], dim=1)
        return e
        
    def forward(self, v, data, weights=None):
        vres = v
        for layer in range(self.n_layers):
            vsrc = (v if self.dynamic_efeats == 'prev' 
                    else data.pos if self.dynamic_efeats == 'orig' 
                    else vres)
            get_extra = not (self.dynamic_efeats == 'first' and layer != 0)
            efeat_fn = self._get_dynamic_edge_feat if self.static_efeats == 0 else self._get_edge_feat

            e = efeat_fn(vsrc, data,
                         rich_efeats=self.rich_efeats and get_extra,
                         euclidian=self.euclidian and get_extra, 
                         direction=self.direction and get_extra,
                         weights=weights if get_extra and self.n_weights > 0 else None)
            v = self.gnn[layer](v, e, data)
        return v + vres if self.residual else v


class Model(nn.Module):
    def __init__(self, num_blocks=9, n_weights=0):
        super().__init__()

        self.in_blocks = nn.ModuleList([
            GNNBlock(feat_dims=[2, 8, 8], bn=True, dp=0.2, static_efeats=2)
        ])
        self.hid_blocks = nn.ModuleList([
            GNNBlock(feat_dims=[8, 8, 8, 8], 
                     efeat_hid_dims=[16],
                     bn=True, 
                     act=True,
                     dp=0.2, 
                     static_efeats=2,
                     dynamic_efeats='skip', 
                     euclidian=True, 
                     direction=True, 
                     n_weights=n_weights,
                     residual=True)
            for _ in range(num_blocks)
        ])
        self.out_blocks = nn.ModuleList([
            GNNBlock(feat_dims=[8, 8], bn=True, static_efeats=2),
            GNNBlock(feat_dims=[8, 2], act=False, static_efeats=2)
        ])
        
        self.normalize = Normalization()

    def forward(self, data, weights=None, output_hidden=False, numpy=False):
        v = data.pos if data.pos is not None else generate_rand_pos(len(data.x)).to(data.x.device)       
        v = self.normalize(v, data)
          
        hidden = []
        for block in chain(self.in_blocks, 
                           self.hid_blocks, 
                           self.out_blocks):
            v = block(v, data, weights)
            if output_hidden:
                hidden.append(v.detach().cpu().numpy() if numpy else v)
        if not output_hidden:
            vout = v.detach().cpu().numpy() if numpy else v
        
        return hidden if output_hidden else vout
    

class GNNGraphDrawing(nn.Module):
    def __init__(self, num_blocks=9, n_weights=0):
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
                     dynamic_efeats='skip', 
                     euclidian=True, 
                     direction=True, 
                     n_weights=n_weights,
                     residual=True)
            for _ in range(num_blocks)
        ])
        self.out_blocks = nn.ModuleList([
            GNNBlock(feat_dims=[8, 8], bn=True),
            GNNBlock(feat_dims=[8, 2], act=False)
        ])
        
        self.normalize = Normalization()

    def forward(self, data, weights=None, output_hidden=False, numpy=False, with_initial_pos=False):
        v = data.x if with_initial_pos else generate_rand_pos(len(data.x)).to(data.x.device)
        v = self.normalize(v, data)

        hidden = []
        for block in chain(self.in_blocks, 
                           self.hid_blocks, 
                           self.out_blocks):
            v = block(v, data, weights)
            if output_hidden:
                hidden.append(v.detach().cpu().numpy() if numpy else v)
        if not output_hidden:
            vout = v.detach().cpu().numpy() if numpy else v

        return hidden if output_hidden else vout
    
    
class DenseLayer(nn.Module):
    ACT = nn.GELU
    
    def __init__(self, in_dim, out_dim=None, skip=True, bn=True, act=True, dp=None, _flip=False):
        super().__init__()
        out_dim = out_dim or in_dim
        if type(act) is bool:
            act = self.ACT() if act else nn.Identity()
        self._flip = _flip
        if self._flip:
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                act,
                nn.BatchNorm1d(out_dim) if bn else nn.Identity(),
                nn.Dropout(dp) if dp is not None else nn.Identity(),
            )
            self.project = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim) if bn else nn.Identity(),
                act,
                nn.Dropout(dp) if dp is not None else nn.Identity(),
            )
            self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self.skip = skip
        
    def forward(self, x):
        if self._flip:
            return self.project(x) + self.net(x) if self.skip else self.net(x)
        return self.proj(x) + self.net(x) if self.skip else self.net(x)

    
class EdgeNet(nn.Module):
    def __init__(self, efeat_dim, nfeat_dims, depth=0, width=None, skip=True, **kwargs):
        super().__init__()
        width = width or efeat_dim
        self.net = nn.Sequential(
            DenseLayer(efeat_dim, width, skip=skip, **kwargs) if depth > 0 else nn.Identity(),
            *[DenseLayer(width, skip=skip, **kwargs) for _ in range(depth - 1)]
        )
        try:
            out_dim = nfeat_dims[0] * nfeat_dims[1]
        except:
            out_dim = nfeat_dims ** 2
        self.out = nn.Linear(width if depth > 0 else efeat_dim, out_dim)

    def forward(self, x):
        return self.out(self.net(x))
    
    
class DepthWiseEdgeNet(nn.Module):
    def __init__(self, efeat_dim, nfeat_dim, depth=0, width=None, skip=True, **kwargs):
        super().__init__()
        width = width or efeat_dim
        self.net = nn.Sequential(
            DenseLayer(efeat_dim, width, skip=skip, **kwargs) if depth > 0 else nn.Identity(),
            *[DenseLayer(width, skip=skip, **kwargs) for _ in range(depth - 1)]
        )
        self.out = nn.Linear(width if depth > 0 else efeat_dim, nfeat_dim)

    def forward(self, x):
        return self.out(self.net(x))
    
    
class Generator(nn.Module):
    def __init__(self, 
                 num_blocks=9, 
                 num_layers=3,
                 num_enet_layers=2,
                 layer_dims=None,
                 n_weights=0, 
                 dynamic_efeats='skip',
                 euclidian=True,
                 direction=True,
                 residual=True,
                 normalize=None):
        super().__init__()

        self.in_blocks = nn.ModuleList([
            GNNBlock(feat_dims=[2, 8, 8 if layer_dims is None else layer_dims[0]], bn=True, dp=0.2, static_efeats=2)
        ])
        self.hid_blocks = nn.ModuleList([
            GNNBlock(feat_dims=layer_dims or ([8] + [8] * num_layers), 
                     efeat_hid_dims=[16] * (num_enet_layers - 1),
                     bn=True, 
                     act=True,
                     dp=0.2, 
                     static_efeats=2,
                     dynamic_efeats=dynamic_efeats,
                     euclidian=euclidian,
                     direction=direction,
                     n_weights=n_weights,
                     residual=residual)
            for _ in range(num_blocks)
        ])
        self.out_blocks = nn.ModuleList([
            GNNBlock(feat_dims=[8 if layer_dims is None else layer_dims[-1], 8], bn=True, static_efeats=2),
            GNNBlock(feat_dims=[8, 2], act=False, static_efeats=2)
        ])
        self.normalize = normalize

    def forward(self, data, weights=None, output_hidden=False, numpy=False):
        v = data.pos if data.pos is not None else generate_rand_pos(len(data.x)).to(data.x.device)
        if self.normalize is not None:
            v = self.normalize(v, data)
        
        hidden = []
        for block in chain(self.in_blocks, 
                           self.hid_blocks, 
                           self.out_blocks):
            v = block(v, data, weights)
            if output_hidden:
                hidden.append(v.detach().cpu().numpy() if numpy else v)
        if not output_hidden:
            vout = v.detach().cpu().numpy() if numpy else v
            if self.normalize is not None:
                vout = self.normalize(vout, data)
        
        return hidden if output_hidden else vout

