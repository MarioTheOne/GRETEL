from src.dataset.data_instance_base import DataInstance
from typing import List

class DataInstanceWFeatures(DataInstance):
    
    
    def __init__(self,
                 id=None,
                 name: str = None,
                 graph=None,
                 graph_dgl=None,
                 graph_label: int = None,
                 node_labels: dict = None,
                 edge_labels: dict = None,
                 mcd: int = None,
                 features: List[float] = None) -> None:
        self._features = features
        super().__init__(id, name, graph,
                         graph_dgl, graph_label,
                         node_labels, edge_labels, mcd)

   
    @property
    def features(self):
        return self._features
    
    @features.setter
    def features(self, new_features):
        self._features = new_features