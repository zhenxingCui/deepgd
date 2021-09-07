import os
import re
import time
import glob
import json
import copy
import shutil
import pickle
import random 
import pathlib
import warnings
import concurrent
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Union
from itertools import chain, combinations, product
from functools import reduce, lru_cache
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import forceatlas2 as fa2
import pydot
import pynvml
from tqdm.notebook import tqdm
from ipywidgets import Tab, Output, HBox, VBox, Layout

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torch.utils.tensorboard import SummaryWriter
from torch import autograd
import torch_scatter
import torch_geometric.utils
import torch_geometric.nn as gnn
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.data.batch import Batch