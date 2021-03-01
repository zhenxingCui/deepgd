import os
import os.path
import re
import time
import glob
import json
import shutil
import pickle
import random 
import pathlib
import warnings
from pprint import pformat
from itertools import chain, combinations
from functools import reduce, lru_cache

import nvidia_smi
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from bentley_ottmann.planar import segments_intersections as intersect

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.utils
import torch_geometric.nn as gnn
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.data.batch import Batch
import pytorch_lightning as pl
