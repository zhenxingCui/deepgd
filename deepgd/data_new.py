from ._dependencies import *

from .tools import *
from .layouts import *
from .functions import *
from .transform import *

import torch_geometric as pyg


# class GraphAttributeModule:
    
#     dependencies = []
    
#     def __init__(self):
#         pass
    
#     def __call__(self):

class RomeDataset(pyg.data.Dataset):
    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root=root, 
                         transform=transform, 
                         pre_transform=pre_transform, 
                         pre_filter=pre_filter)
        