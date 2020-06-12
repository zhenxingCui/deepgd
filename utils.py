import os
import random 
import glob
import json
import pickle
import pathlib
from pprint import pformat
from itertools import chain, combinations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import NNConv, BatchNorm
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data.batch import Batch
import warnings
from tqdm.notebook import tqdm
import nvidia_smi
from bentley_ottmann.planar import segments_intersections as intersect

class Config:
    class Store:
        def __init__(self, data: dict):
            def wrap(kvpair):
                key, value = kvpair
                if type(value) is dict:
                    return key, Config.Store(data=value)
                return kvpair
            self.__dict__ = dict(map(wrap, data.items()))

        def __getitem__(self, item):
            if item is ...:
                return self.__dict__
            return self.__dict__[item]

        def __repr__(self):
            return pformat(self.__dict__)#, sort_dicts=False)

        def __str__(self):
            return str(self.__dict__)
        
    def __init__(self, file):
        self.file = file
        
    def __getitem__(self, item):
        data = Config.Store(json.load(open(self.file)))
        if item is ...:
            return data
        return data[item]
    
    def __getattr__(self, attr):
        return self[attr]
    
    def __repr__(self):
        return pformat(self[...])#, sort_dicts=False)
    
    def __str__(self):
        return str(self[...])
    
    
def node2edge(node_pos, batch):
    return node_pos[batch.edge_index[0, :].t()], node_pos[batch.edge_index[1, :].t()]


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
    return [(i, j) for i in range(size) for j in range(size) if i != j]


def find_intersect(segments,accurate = True):
    intersect(segments)
    
def get_euclidian(node_feat, batch):
    eps = 1e-5
    start_pos, end_pos = node2edge(node_feat, batch)
    vector = end_pos - start_pos
    euclidian = vector.norm(2, dim=1).unsqueeze(dim=1)
    direction = vector / (euclidian + eps)
    return euclidian, direction


def get_edge_feat(v, data, euclidian=False, direction=False):
        e = data.edge_attr
        if euclidian or direction:
            u, d = get_euclidian(v, data)
            if euclidian:
                e = torch.cat([e, u], dim=1)
            if direction:
                e = torch.cat([e, d], dim=1)
        return e
    
    
def generate_eAttr(G, com_edge_list):
    path_length = dict(nx.all_pairs_shortest_path_length(G))
    max_length = 0
    for source in path_length:
        for target in path_length[source]:
            if path_length[source][target] > max_length:
                max_length = path_length[source][target]
    L = 1 
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

    
def load_processed_data(G_list_file='G_list.pickle', 
                        data_list_file='data_list.pickle', 
                        index_file='data_index.txt'):
    if os.path.isfile(G_list_file) and os.path.isfile(data_list_file):
        G_list = pickle.load(open(G_list_file, 'rb'))
        data_list = pickle.load(open(data_list_file, 'rb'))
    else:
        if not os.path.isfile(index_file):
            shuffle_rome(index_file)
        rome = load_rome(index_file)
        G_list, data_list = convert_datalist(rome)
        pickle.dump(G_list, open(G_list_file, 'wb'))
        pickle.dump(data_list, open(data_list_file, 'wb'))
    return G_list, data_list


def train(model, criterion, optimizer, loader, device, progress=None, cuda=None, output_components=False):
    model.train()
    loss_all = []
    graphs_all = 0
    if output_components:
        components_all = []
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        if output_components:
            loss, components = criterion(output, data,output_components=True)
        else:
            loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        loss_all.append(loss.item())
        graphs_all += data.num_graphs
        if output_components:
            components_all.append([comp.detach().cpu().numpy() for comp in components])
        if progress is not None:
            postfix = {'loss': loss.item() / data.num_graphs}
            if cuda is not None:
                util_info = nvidia_smi.nvmlDeviceGetUtilizationRates(cuda)
                mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(cuda)
                postfix['gpu%'] = util_info.gpu
                postfix['mem%'] = mem_info.used / mem_info.total * 100
            progress.update()
            progress.set_postfix(postfix)
    if output_components:
        return np.sum(loss_all) / graphs_all, np.sum(components_all, axis=0) / graphs_all
    else:
        return np.sum(loss_all) / graphs_all
    

def evaluate(model, data, criterion, device, output_components=False, reduction=None, with_initial_pos=False):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        hidden = model(data, output_hidden=True, with_initial_pos=with_initial_pos)
        if reduction is None:
            pred = hidden[-1]  
        else:
            pred = torch.tensor(reduction(hidden[-2].detach().cpu().numpy()), device=data.x.device)
        if output_components:
            loss, components = criterion(pred, data, output_components=True)
            loss = loss.cpu().numpy()
            components = list(map(lambda x: x.cpu().numpy(), components))
        else:
            loss = criterion(pred, data).cpu().numpy()
        loss = round(float(loss),2)
    if output_components:
        return pred.cpu().numpy(), loss, components
    else:
        return pred.cpu().numpy(), loss


def shuffle_rome(index_file):
    files = glob.glob('../rome/*.graphml')
    random.shuffle(files)
    with open(index_file, "w") as fout:
        for f in files:
            print(f, file=fout)

            
def load_rome(index_file):
    G_list = []
    count = 0
    for file in tqdm(open(index_file).read().splitlines(), desc="load_rome"):
        G = nx.read_graphml(file)
        G_list.append(G)
    return G_list


def get_ground_truth(G, prog='neato'):
    return np.array(list(nx.nx_agraph.graphviz_layout(G, prog=prog).values()))


def graph_vis(G, node_pos, file_name=None, **kwargs):
    graph_attr = dict(node_size=100, 
                      with_labels=False, 
                      labels=dict(zip(list(G.nodes), map(lambda n: n[1:], list(G.nodes)))),
                      font_color="white", 
                      font_weight="bold",
                      font_size=12)
    graph_attr.update(kwargs)
    for i, (n, p) in enumerate(node_pos):
        G.nodes[f'n{i}']['pos'] = n, p
    pos = nx.get_node_attributes(G, name='pos')
    plt.figure()
    nx.draw(G, pos, **graph_attr)
    if file_name is not None:
        plt.savefig(file_name) 
    
    
def convert_datalist(rome):
    data_list = []
    G_list = []
    for G in tqdm(rome, desc="convert_datalist"):
        size = G.number_of_nodes()
        com_edge_list = generate_edgelist(size)
        try:
            edge_attr = generate_eAttr(G, com_edge_list)
        except KeyError:
            continue
        except ZeroDivisionError:
            continue
        edge_index = torch.tensor(com_edge_list, dtype=torch.long)
        nodes_degree = [i[1] for i in list(G.degree(list(G.nodes)))]
        x = torch.tensor(nodes_degree,dtype=torch.float)
        data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
        data_list.append(data)
        G_list.append(G)
    return G_list, data_list


def get_stress_minimizing_scale(pos, batch):
    d, w = batch.edge_attr[:, 0].numpy(), batch.edge_attr[:, 1].numpy()
    start, end = node2edge(pos, batch)
    diff = end - start
    dist = np.linalg.norm(diff, axis=1)
    return (w * d * dist).sum() / (w * dist * dist).sum()


def pca_project(vector, n=2):
    if vector.shape[1] == n:
        return vector
    pca = PCA(n_components=n)
    return pca.fit_transform(vector)


def tsne_project(vector, n=2):
    tsne = TSNE(n_components=n)
    return tsne.fit_transform(vector)


def umap_project(vector, n=2, **kwargs):
    umap = UMAP(n_components=n, **kwargs)
    return umap.fit_transform(vector)