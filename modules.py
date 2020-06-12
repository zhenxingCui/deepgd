import os
import random 
import glob
import json
import pickle
import pathlib
from itertools import chain, combinations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.data.batch import Batch
import networkx as nx
import warnings
from tqdm.notebook import tqdm
import nvidia_smi
from sklearn import preprocessing
from utils import *


def normalize(edge_vectors):
    return edge_vectors / (np.linalg.norm(edge_vectors, axis=1, keepdims=True) + 1e-5)


def torch_normalize(x):
    return x / (x.norm(dim=1).unsqueeze(dim=1) + 1e-5)


def get_real_edges(batch):
    l = batch.edge_attr[:, 0]
    edges = batch.edge_index.T
    return edges[l == l.min()]


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


def get_theta_angles_and_node_degrees(node_pos, batch, return_u=False):
    real_edges = get_real_edges(batch)
    angles = get_counter_clockwise_sorted_angle_vertices(real_edges, node_pos)
    u, v1, v2 = angles[:, 0], angles[:, 1], angles[:, 2]
    e1 = torch_normalize(node_pos[v1] - node_pos[u])
    e2 = torch_normalize(node_pos[v2] - node_pos[u])
    theta = (e1 * e2).sum(dim=1).acos()
    degrees = get_node_degrees_by_indices(real_edges, u)
    if return_u:
        return theta, degrees, u
    else:
        return theta, degrees


# min over u
def resolution_score(theta, degree, u):
    sorted_idx = sorted(np.arange(len(u)), key=lambda e: (u[e], theta[e]))
    u_idx = np.unique(u, return_index=True)[1]
    min_theta = theta[sorted_idx][u_idx].cpu()
    min_degree = degree[u_idx].cpu()
    return (min_theta * min_degree/(2 * np.pi)).mean().item()
    

def get_node_degrees_by_indices(real_edges, indices):
    node, degrees = np.unique(real_edges[:, 0].detach().cpu().numpy(), return_counts=True)
    return torch.tensor(degrees[indices], device=real_edges.device)


class EnergyLossVectorized(torch.nn.Module): # StressLoss
    def __init__(self):
        super().__init__()
        
    def forward(self, p, batch):
        edge_attr = batch.edge_attr
        # convert per-node positions to per-edge positions
        start, end = node2edge(p, batch)
        
        start_x = start[:, 0]
        start_y = start[:, 1]
        end_x = end[:, 0]
        end_y = end[:, 1]
        
        l = edge_attr[:, 0]
        k = edge_attr[:, 1]
        
        term1 = (start_x - end_x).pow(2)
        term2 = (start_y - end_y).pow(2)
        term3 = l.pow(2)
        term4 = 2 * l * (term1 + term2).abs().sqrt()
        energy = k / 2 * (term1 + term2 + term3 - term4)
        return energy.sum()


# NEW #
class MeanStressLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, p, batch):
        edge_attr = batch.edge_attr
        # convert per-node positions to per-edge positions
        start, end = node2edge(p, batch)
        
        start_x = start[:, 0]
        start_y = start[:, 1]
        end_x = end[:, 0]
        end_y = end[:, 1]
        
        l = edge_attr[:, 0]
        k = edge_attr[:, 1]
        
        term1 = (start_x - end_x).pow(2)
        term2 = (start_y - end_y).pow(2)
        term3 = l.pow(2)
        term4 = 2 * l * (term1 + term2).abs().sqrt()
        energy = k / 2 * (term1 + term2 + term3 - term4)
        return energy.mean()
    
    
class AngularResolutionLoss(nn.Module): # AngularResolutionSAELoss
    def __init__(self):
        super().__init__()
        
    def forward(self, node_pos, batch):
        theta, degrees = get_theta_angles_and_node_degrees(node_pos, batch)
        phi = degrees.float().pow(-1).mul(2*np.pi)
        return phi.sub(theta).abs().sum()
    

class AngularResolutionL2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, node_pos, batch):
        theta, degrees = get_theta_angles_and_node_degrees(node_pos, batch)
        phi = degrees.float().pow(-1).mul(2*np.pi)
        return phi.sub(theta).square().sum().abs().sqrt()
    
    
# NEW #
class AngularResolutionLinfLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, node_pos, batch):
        theta, degrees = get_theta_angles_and_node_degrees(node_pos, batch)
        phi = degrees.float().pow(-1).mul(2*np.pi)
        return phi.sub(theta).max()
    
    
# NEW #
class AngularResolutionMAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, node_pos, batch):
        theta, degrees = get_theta_angles_and_node_degrees(node_pos, batch)
        phi = degrees.float().pow(-1).mul(2*np.pi)
        return phi.sub(theta).abs().mean()
    

# NEW #
class AngularResolutionMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, node_pos, batch):
        theta, degrees = get_theta_angles_and_node_degrees(node_pos, batch)
        phi = degrees.float().pow(-1).mul(2*np.pi)
        return phi.sub(theta).square().mean()
    
    
class AngularResolutionSquareLoss(nn.Module): # AngularResolutionSSELoss
    def __init__(self):
        super().__init__()
        
    def forward(self, node_pos, batch):
        theta, degrees = get_theta_angles_and_node_degrees(node_pos, batch)
        phi = degrees.float().pow(-1).mul(2*np.pi)
        return phi.sub(theta).square().sum()
    
    
class AngularResolutionRatioLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, node_pos, batch):
        eps = 1e-5
        theta, degrees = get_theta_angles_and_node_degrees(node_pos, batch)
        phi = degrees.float().pow(-1).mul(2*np.pi)
        return phi.sub(theta).abs().div(theta + eps).sum()
    
    
# NEW #
class AngularResolutionMeanRatioLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, node_pos, batch):
        eps = 1e-5
        theta, degrees = get_theta_angles_and_node_degrees(node_pos, batch)
        phi = degrees.float().pow(-1).mul(2*np.pi)
        return phi.sub(theta).abs().div(theta + eps).mean()
    
    
class AngularResolutionSineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, node_pos, batch):
        theta, degrees = get_theta_angles_and_node_degrees(node_pos, batch)
        phi = degrees.float().pow(-1).mul(2*np.pi)
        return phi.sub(theta).abs().div(2).sin().sum()
    
    
# NEW #
class AngularResolutionMeanSineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, node_pos, batch):
        theta, degrees = get_theta_angles_and_node_degrees(node_pos, batch)
        phi = degrees.float().pow(-1).mul(2*np.pi)
        return phi.sub(theta).abs().div(2).sin().mean()
    
    
class RingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, node_pos, batch):
        eps = 1e-5
        edge_attr = batch.edge_attr
        # convert per-node positions to per-edge positions
        start, end = node2edge(node_pos, batch)
        
        start_x = start[:, 0]
        start_y = start[:, 1]
        end_x = end[:, 0]
        end_y = end[:, 1]
        
        dist = (end - start).norm(dim=1)
        
        l = edge_attr[:, 0].min()
        k = 1
        
        energy = l * k / (dist + eps)
        return energy.sum()
    
    
# NEW #
class MeanRingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, node_pos, batch):
        eps = 1e-5
        edge_attr = batch.edge_attr
        # convert per-node positions to per-edge positions
        start, end = node2edge(node_pos, batch)
        
        start_x = start[:, 0]
        start_y = start[:, 1]
        end_x = end[:, 0]
        end_y = end[:, 1]
        
        dist = (end - start).norm(dim=1)
        
        l = edge_attr[:, 0].min()
        k = 1
        
        energy = l * k / (dist + eps)
        return energy.mean()
    
    
class CompositeLoss(nn.Module):
    def __init__(self, criterions, weights=None):
        super().__init__()
        self.weights = np.ones(len(criterions)) if weights is None else weights
        self.criterions = criterions
        
    def __len__(self):
        return len(self.criterions)
        
    def forward(self, *args, output_components=False, **kwargs):
        losses = 0
        components = []
        for criterion, weight in zip(self.criterions, self.weights):
            loss = criterion(*args, **kwargs)
            losses += loss * weight
            components += [loss]
        if output_components:
            return losses, components
        else:
            return losses
    

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
        
    def forward(self, v, data):
        vres = v
        for layer in range(self.n_layers):
            vsrc = v if self.euclidian == 'prev' else vres
            get_extra = not (self.extra_efeat == 'first' and layer != 0)
            e = get_edge_feat(vsrc, data, 
                              euclidian=self.euclidian and get_extra, 
                              direction=self.direction and get_extra)
            v = self.gnn[layer](v, e, data)
        return v + vres if self.residual else v
        