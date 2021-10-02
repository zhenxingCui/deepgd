from typing import Tuple

from ._dependencies import *

from .tools import *
from .layouts import *
from .functions import *
from .transform import *

from typing import TypeVar
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch_geometric as pyg


class GraphDataset:
    pass

class Attribute(pyg.data.Dataset, ABC):
    @dataclass
    class _Record:
        cls: type
        dependencies: list[str]

    _registry: dict[str, _Record] = {}

    def __init_subclass__(cls, name, dependencies=None, **kwargs):
        super.__init_subclass__(**kwargs)
        cls._registry[name] = cls._Record(cls, dependencies or [])

    def __new__(cls, name, **kwargs):
        return object.__new__(cls._registry[name])

    def __init__(self, name, *, dataset: GraphDataset, dependencies=None, root=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root=root, 
                         transform=transform, 
                         pre_transform=pre_transform, 
                         pre_filter=pre_filter)
        self.name = name
        self.dataset = dataset
        self.dependencies = dependencies

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        raise NotImplemented

    @property
    @abstractmethod
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        raise NotImplemented

    def process(self):
        pass

    def len(self) -> int:
        pass

    def get(self, idx: int) -> Data:
        pass

    @abstractmethod
    def generate(self, ):


class RawEdgeIndex(Attribute, name='raw_edge_index', dependencies=[]):
    pass