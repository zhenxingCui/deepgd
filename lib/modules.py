from .imports import *
from .functions import *


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
        
        
class GNNLayer(nn.Module):
    def __init__(self, 
                 in_vfeat, 
                 out_vfeat, 
                 in_efeat, 
                 edge_net=None, 
                 bn=False, 
                 act=False, 
                 dp=None, 
                 aggr='mean',
                 root_weight=True):
        super().__init__()
        self.enet = nn.Linear(in_efeat, in_vfeat * out_vfeat) if edge_net is None and in_efeat > 0 else edge_net
        self.conv = gnn.NNConv(in_vfeat, out_vfeat, nn=self.enet, aggr=aggr, root_weight=root_weight)
        self.bn = gnn.BatchNorm(out_vfeat) if bn else nn.Identity()
        self.act = nn.LeakyReLU() if act else nn.Identity()
        self.dp = dp and nn.Dropout(dp) or nn.Identity()
        
    def forward(self, v, e, data):
        v = self.conv(v, data.edge_index, e)
        v = self.bn(v)
        v = self.act(v)
        v = self.dp(v)
        return v


class GNNBlock(nn.Module):
    def __init__(self, 
                 feat_dims, 
                 efeat_hid_dims=[], 
                 efeat_hid_acts=nn.LeakyReLU,
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
                                           [efeat_hid_acts] * len(efeat_hid_dims) + [nn.Tanh])
            ))
            self.gnn.append(GNNLayer(in_vfeat=in_feat, 
                                     out_vfeat=out_feat, 
                                     in_efeat=in_efeat_dim, 
                                     edge_net=edge_net,
                                     bn=bn, 
                                     act=act, 
                                     dp=dp,
                                     aggr=aggr,
                                     root_weight=root_weight))
        
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

    def forward(self, data, weights=None, output_hidden=False, numpy=False):
        v = data.pos if data.pos is not None else generate_rand_pos(len(data.x)).to(data.x.device)       
        v = rescale_with_minimized_stress(v, data)
          
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
    
    
# class GNNBlockForEdgePrediction(nn.Module):
#     def __init__(self, feat_dims, 
#                  efeat_hid_dims=[], 
#                  efeat_hid_acts=nn.LeakyReLU,
#                  bn=False, 
#                  act=True, 
#                  dp=None, 
#                  n_edge_attr=0,
#                  extra_efeat='skip',
#                  euclidian=False, 
#                  direction=False,
#                  residual=False):
#         '''
#         extra_efeat: {'skip', 'first', 'prev'}
#         '''
#         super().__init__()
        
#         self.n_edge_attr=n_edge_attr
#         self.extra_efeat = extra_efeat
#         self.euclidian = euclidian
#         self.direction = direction
#         self.residual = residual
#         self.gnn = nn.ModuleList()
#         self.n_layers = len(feat_dims) - 1
        
#         for idx, (in_feat, out_feat) in enumerate(zip(feat_dims[:-1], feat_dims[1:])):
#             direction_dim = feat_dims[idx] if self.extra_efeat == 'prev' else feat_dims[0]
#             in_efeat_dim = self.n_edge_attr
#             if self.extra_efeat != 'first': 
#                 in_efeat_dim += self.euclidian + self.direction * direction_dim
#             edge_net = nn.Sequential(*chain.from_iterable(
#                 [nn.Linear(idim, odim),
#                  nn.BatchNorm1d(odim),
#                  act()]
#                 for idim, odim, act in zip([in_efeat_dim] + efeat_hid_dims,
#                                            efeat_hid_dims + [in_feat * out_feat],
#                                            [efeat_hid_acts] * len(efeat_hid_dims) + [nn.Tanh])
#             )) if in_efeat_dim > 0 else None
#             self.gnn.append(GNNLayer(in_vfeat=in_feat, 
#                                      out_vfeat=out_feat, 
#                                      in_efeat=in_efeat_dim, 
#                                      edge_net=edge_net,
#                                      bn=bn, 
#                                      act=act, 
#                                      dp=dp))
        
#     def _get_edge_feat(self, pos, data, euclidian=False, direction=False):
#         e = torch.zeros(len(data.model_edge_index), 0) if data.model_edge_attr is None else data.model_edge_attr
#         if euclidian or direction:
#             start_pos, end_pos = get_full_edges(pos, data)
#             d, u = l2_normalize(end_pos - start_pos, return_norm=True)
#             if euclidian:
#                 e = torch.cat([e, u], dim=1)
#             if direction:
#                 e = torch.cat([e, d], dim=1)
#         return e if e.numel() > 0 else None
        
#     def forward(self, v, data):
#         vres = v
#         for layer in range(self.n_layers):
#             vsrc = v if self.euclidian == 'prev' else vres
#             get_extra = not (self.extra_efeat == 'first' and layer != 0)
            
#             e = self._get_edge_feat(vsrc, data, 
#                                     euclidian=self.euclidian and get_extra, 
#                                     direction=self.direction and get_extra)
#             v = self.gnn[layer](v, e, data)
#         return v + vres if self.residual else v
    

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

    def forward(self, data, weights=None, output_hidden=False, numpy=False, with_initial_pos=False):
        v = data.x if with_initial_pos else generate_rand_pos(len(data.x)).to(data.x.device)
        v = rescale_with_minimized_stress(v, data)

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
        
        
class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False, act=False, dp=None):
        super().__init__()
        self.dense = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        if type(act) is bool:
            self.act = nn.LeakyReLU() if act else nn.Identity()
        else:
            self.act = act
        self.dp = nn.Dropout(dp) if dp is not None else nn.Identity()
        
    def forward(self, x):
        x = self.dense(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dp(x)
        return x
        

class EdgeFeatureDiscriminator(nn.Module):
    def __init__(self, softplus=True):
        super().__init__()
        self.conv = GNNBlock(feat_dims=[2, 8, 8, 16, 16, 32], bn=True, dp=0.2, static_efeats=2)
        self.dense1 = DenseLayer(in_channels=32, out_channels=16, bn=False, act=True, dp=0.3)
        self.dense2 = DenseLayer(in_channels=16, out_channels=8, bn=False, act=True, dp=0.3)
        self.dense3 = DenseLayer(in_channels=8, out_channels=1, bn=False, act=nn.Softplus() if softplus else False, dp=None)
        
    def forward(self, batch):
        x = self.conv(batch.pos, batch)
        feats = gnn.global_mean_pool(x, batch.batch)
        x = self.dense1(feats)
        x = self.dense2(x)
        x = self.dense3(x)
        return x.flatten()
        
        
class Discriminator(nn.Module):
    def __init__(self, conv=5, dense=3, softplus=True):
        super().__init__()
        self.n_conv = conv;
        self.n_dense = dense;
        self.conv1 = GCNLayer(in_channels=2, out_channels=8, bn=False, act=True, dp=None)
        self.conv2 = GCNLayer(in_channels=8, out_channels=16, bn=False, act=True, dp=0.1)
        self.conv3 = GCNLayer(in_channels=16, out_channels=32, bn=False, act=True, dp=0.1)
        if self.n_conv == 4:
            self.conv4 = GCNLayer(in_channels=32, out_channels=128, bn=False, act=False, dp=None)
        elif self.n_conv == 5:
            self.conv4 = GCNLayer(in_channels=32, out_channels=64, bn=False, act=True, dp=0.1)
            self.conv5 = GCNLayer(in_channels=64, out_channels=128, bn=False, act=False, dp=None)
        elif self.n_conv == 6:
            self.conv4 = GCNLayer(in_channels=32, out_channels=32, bn=False, act=True, dp=0.1)
            self.conv5 = GCNLayer(in_channels=32, out_channels=64, bn=False, act=True, dp=0.1)
            self.conv6 = GCNLayer(in_channels=64, out_channels=128, bn=False, act=False, dp=None)
        if self.n_dense == 3:
            self.dense1 = DenseLayer(in_channels=128, out_channels=32, bn=False, act=True, dp=0.3)
            self.dense2 = DenseLayer(in_channels=32, out_channels=8, bn=False, act=True, dp=0.3)
            self.dense3 = DenseLayer(in_channels=8, out_channels=1, bn=False, act=nn.Softplus() if softplus else False, dp=None)
        elif self.n_dense == 4:
            self.dense1 = DenseLayer(in_channels=128, out_channels=64, bn=False, act=True, dp=0.3)
            self.dense2 = DenseLayer(in_channels=64, out_channels=32, bn=False, act=True, dp=0.3)
            self.dense3 = DenseLayer(in_channels=32, out_channels=8, bn=False, act=True, dp=0.3)
            self.dense4 = DenseLayer(in_channels=8, out_channels=1, bn=False, act=nn.Softplus() if softplus else False, dp=None)
        
    def forward(self, batch):
        x = self.conv1(batch.pos, batch)
        x = self.conv2(x, batch)
        x = self.conv3(x, batch)
        x = self.conv4(x, batch)
        if self.n_conv >= 5:
            x = self.conv5(x, batch)
        if self.n_conv >= 6:
            x = self.conv6(x, batch)
        feats = gnn.global_mean_pool(x, batch.batch)
        x = self.dense1(feats)
        x = self.dense2(x)
        x = self.dense3(x)
        if self.n_dense >= 4:
            x = self.dense4(x)
        
        return x.flatten()
    
    
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
                 residual=True):
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

    def forward(self, data, weights=None, output_hidden=False, numpy=False):
        v = data.pos if data.pos is not None else generate_rand_pos(len(data.x)).to(data.x.device)       
        v = rescale_with_minimized_stress(v, data)
        
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
    
