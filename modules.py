import os
import random 
import glob
import json
import pickle
import pathlib
from itertools import chain, combinations
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import NNConv, BatchNorm
import networkx as nx
from matplotlib import pyplot as plt
from torch_geometric.data.batch import Batch
import warnings
from tqdm.notebook import tqdm
from torch.nn import Dropout


class Config:
    def __init__(self, file):
        self.file = file
        
    def __getitem__(self, index):
        data = json.load(open(self.file))
        if index is ...:
            return data
        return data[index]

def generate_polygon(n, radius=1):
    node_pos = [(radius * np.cos(2 * np.pi * i / n),
                 radius * np.sin(2 * np.pi * i / n)) for i in range(n)]
    x = torch.tensor(node_pos,dtype=torch.float)
    return x

def generate_randPos(n, seed=0):
    random.seed(seed)
    node_pos = [(random.uniform(-1, 1),
                 random.uniform(-1, 1)) for i in range(n)]
    x = torch.tensor(node_pos,dtype=torch.float)
    return x

def generate_edgelist(size):
    edge_list = []
    for i in range(size):
        for j in range(size):
#             p = random.random()
#             if p > 0.5:
#                 edge_list.append((i,j))
#             else:
#                 edge_list.append((j,i))  
            if i != j:
                edge_list.append((i,j))
    return edge_list

def node2edge(node_pos, batch):
    data_list = batch.to_data_list() if type(batch) is Batch else [batch]
    # find sizes for each graph in batch
    graph_sizes = list(map(lambda d: d.x.size()[0], data_list))
    # get split indices
    start_idx = np.insert(np.cumsum(graph_sizes), 0, 0)
    start_pos = []
    end_pos = []
    n_nodes = []
    for i, num_nodes in enumerate(graph_sizes):
        # get edge list for current graph
        edgelist = np.array(generate_edgelist(num_nodes))
        # get node positions for current graph
        graph_node_pos = node_pos[start_idx[i]:start_idx[i+1]]
        # get edge start positions for current graph
        start_pos += [graph_node_pos[edgelist[:, 0]]]
        # get edge end positions for current graph
        end_pos += [graph_node_pos[edgelist[:, 1]]]
        n_nodes += [torch.tensor([num_nodes] * len(edgelist),dtype=torch.float)]
    # concatenate the results
    return torch.cat(start_pos, 0), torch.cat(end_pos, 0), torch.cat(n_nodes, 0)

def generate_eAttr(G, com_edge_list):
    path_length = dict(nx.all_pairs_shortest_path_length(G))
    max_length = 0
    for source in path_length:
        for target in path_length[source]:
            if path_length[source][target] > max_length:
                max_length = path_length[source][target]
    L = 2/max_length
    K = 1
    edge_attr = []
    for i in com_edge_list:
        start = "n" + str(i[0])
        end = "n" + str(i[1])
        d = path_length[start][end]
        l = L * d #l = L * d
        k = K/(d**2) 
        start_degree = G.degree(start)
#         end_degree = G.degree(end)
        edge_attr.append([l,k])
    out = torch.tensor(edge_attr, dtype=torch.float)
    return out

def generate_graph(size):
    while True:
        G = nx.binomial_graph(size, random.uniform(0,0.2),directed=False)
#         G = nx.random_powerlaw_tree(size,3,tries=10000)
        com_edge_list = generate_edgelist(size)
        try:
            edge_attr = generate_eAttr(G, com_edge_list)
        except KeyError:
            continue
        except ZeroDivisionError:
            continue
#         nx.write_edgelist(G, file_name, data=False)
        edge_index = torch.tensor(com_edge_list, dtype=torch.long)
        x = generate_randPos(size)
        data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
        return G, data

def generate_testgraph(size,prob):
    while True:
        G = nx.binomial_graph(size, prob,directed=False)
#         G = nx.random_powerlaw_tree(size,3,tries=10000)
        com_edge_list = generate_edgelist(size)
        try:
            edge_attr = generate_eAttr(G, com_edge_list)
        except KeyError:
            continue
        except ZeroDivisionError:
            continue
#         nx.write_edgelist(G, file_name, data=False)
        edge_index = torch.tensor(com_edge_list, dtype=torch.long)
        x = generate_randPos(size)
        data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
        return G, data


class EnergyLossVectorized(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, p, data):
        edge_attr = data.edge_attr
        # convert per-node positions to per-edge positions
        start, end, n_nodes = node2edge(p, data)
        
        start_x = start[:, 0]
        start_y = start[:, 1]
        end_x = end[:, 0]
        end_y = end[:, 1]
        
        l = edge_attr[:, 0]
        k = edge_attr[:, 1]
        
        term1 = (start_x - end_x) ** 2
        term2 = (start_y - end_y) ** 2
        term3 = l ** 2
        term4 = 2 * l * (term1 + term2).sqrt()
        energy = k / 2 * (term1 + term2 + term3 - term4)
        return energy.sum()
    
class CosineAngleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def _get_real_edges(self, batch):
        data_list = batch.to_data_list() if type(batch) is Batch else [batch]
    
        offset = 0
        neighbor_mask_, edge_list_ = [], []
        for data in data_list:
            size = data.num_nodes
            edge_list_.append(np.array(generate_edgelist(size)) + offset)
            l = data.edge_attr[:, 0].detach().cpu().numpy()
            neighbor_mask_.append(l == l.min())
            offset += size
        neighbor_mask = np.concatenate(neighbor_mask_)
        edge_list = np.concatenate(edge_list_)

        return edge_list[neighbor_mask]
        
    def _get_angles(self, real_edges):
        vi0_, vi12_ = [], []
        for i in np.unique(real_edges[:, 0]):
            vi = list(combinations(real_edges[real_edges[:, 0] == i][:, 1], 2))
            vi0_.append(np.repeat(i, len(vi)))
            vi12_.append(np.array(vi).reshape((-1, 2)))
        vi0 = np.concatenate(vi0_)
        vi1, vi2 = np.concatenate(vi12_).astype(int).T
        
        return vi0, vi1, vi2
    
    def _normalize(self, edge_vectors):
        return edge_vectors / ((edge_vectors ** 2).sum(dim=1)
                                                  .sqrt()
                                                  .repeat(2, 1)
                                                  .t())
        
    def forward(self, node_pos, batch):
        real_edges = self._get_real_edges(batch)
        vi0, vi1, vi2 = self._get_angles(real_edges)
    
        e1 = self._normalize(node_pos[vi1] - node_pos[vi0])
        e2 = self._normalize(node_pos[vi2] - node_pos[vi0])
        
        return (e1 * e2).sum()
    
class CompositeLoss(nn.Module):
    def __init__(self, criterions, weights=None):
        super().__init__()
        self.weights = np.ones(len(criterions)) if weights is None else weights
        self.criterions = criterions
        
    def forward(self, *args, **kwargs):
        losses = 0
        for criterion, weight in zip(self.criterions, self.weights):
            losses += criterion(*args, **kwargs) * weight
        return losses
    
def train(model, criterion, optimizer, loader, data_list, device, epoch=None):
    model.train()
    loss_all = 0
    desc = f"epoch {epoch}" if epoch is not None else None
    for data in tqdm(loader, desc=desc, leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(data_list)

def evaluate(model,data,criterion,device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        pred = model(data).detach()
        loss = criterion(pred,data).cpu().numpy()
        loss = round(float(loss),2)
    return pred.cpu().numpy(),loss

def shuffle_rome(index_file):
    files = glob.glob('../rome/*.graphml')
    random.shuffle(files)
    with open(index_file, "w") as fout:
        for f in files:
            print(f, file=fout)

def load_rome(index_file):
    G_list = []
    count = 0
    for file in open(index_file).read().splitlines():
        G = nx.read_graphml(file)
        G_list.append(G)
    return G_list

def graph_vis(G, node_pos, file_name):
    i = 0
    for n, p in node_pos:
        node = 'n' +str(i)
        G.nodes[node]['pos'] = (n,p)
        i += 1
    pos = nx.get_node_attributes(G,'pos')
    plt.figure()
    nx.draw(G, pos)
    plt.savefig(file_name) 
    
def convert_datalist(rome):
    data_list = []
    G_list = []
    for G in rome:
        size = G.number_of_nodes()
        com_edge_list = generate_edgelist(size)
        try:
            edge_attr = generate_eAttr(G, com_edge_list)
        except KeyError:
            continue
        except ZeroDivisionError:
            continue
        edge_index = torch.tensor(com_edge_list, dtype=torch.long)
        x = generate_randPos(size)
        data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
        data_list.append(data)
        G_list.append(G)
    return G_list,data_list


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        edge_feats = 2
        
        self.conv1 = NNConv(2, 16, Linear(edge_feats, 2*16), aggr='mean')
        self.conv2 = NNConv(16, 16, Linear(edge_feats, 16*16), aggr='mean')
        self.conv3 = NNConv(4*16, 2, Linear(edge_feats, 4*16*2), aggr='mean')
        self.relu = nn.LeakyReLU()
        self.bn1 = BatchNorm(16)
        self.bn2 = BatchNorm(16)
        self.bn3 = BatchNorm(16)
        self.bn4 = BatchNorm(16)
#         self.conv4 = NNConv(16+32+64, 128, Linear(2, (16+32+64)*128))
#         self.conv5 = NNConv(128,2,Linear(2,128*2))
#         self.conv2 = NNConv(16,2,Linear(2,16*2))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x1 = self.relu(self.conv1(x, edge_index, edge_attr))
        x2 = self.relu(self.conv2(x1, edge_index, edge_attr))
        x3 = self.relu(self.conv2(x2, edge_index, edge_attr))
        x4 = self.relu(self.conv2(x3, edge_index, edge_attr))
        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        x6 = self.conv3(x5, edge_index, edge_attr)
#         x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
#         x3 = F.relu(self.conv3(x2, edge_index, edge_attr))
        
#         x4 = F.relu(self.conv4(x, edge_index, edge_attr))
#         x5 = F.relu(self.conv5(x4,edge_index,edge_attr))
        return x6

class Net_WD_BN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        edge_feats = 3
        
        self.conv1 = NNConv(2, 16, Linear(edge_feats, 2*16), aggr='mean')
        self.conv2 = NNConv(16, 16, Linear(edge_feats, 16*16), aggr='mean')
        self.conv3 = NNConv(4*16, 2, Linear(edge_feats, 4*16*2), aggr='mean')
        self.relu = nn.LeakyReLU()
        self.bn1 = BatchNorm(16)
        self.bn2 = BatchNorm(16)
        self.bn3 = BatchNorm(16)
        self.bn4 = BatchNorm(16)
#         self.conv4 = NNConv(16+32+64, 128, Linear(2, (16+32+64)*128))
#         self.conv5 = NNConv(128,2,Linear(2,128*2))
#         self.conv2 = NNConv(16,2,Linear(2,16*2))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x1 = self.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x2 = self.relu(self.bn2(self.conv2(x1, edge_index, edge_attr)))
        x3 = self.relu(self.bn3(self.conv2(x2, edge_index, edge_attr)))
        x4 = self.relu(self.bn4(self.conv2(x3, edge_index, edge_attr)))
        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        x6 = self.conv3(x5, edge_index, edge_attr)
#         x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
#         x3 = F.relu(self.conv3(x2, edge_index, edge_attr))
        
#         x4 = F.relu(self.conv4(x, edge_index, edge_attr))
#         x5 = F.relu(self.conv5(x4,edge_index,edge_attr))
        return x6
