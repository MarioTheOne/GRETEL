from copy import deepcopy
#from types import NoneType

import networkx as nx
import numpy as np

from src.n_dataset.instances.base import DataInstance


class GraphInstance(DataInstance):

    def __init__(self, id, label, data, node_features=None, edge_features=None, edge_weights=None, graph_features=None, dataset=None):
        super().__init__(id, label, data, dataset=dataset)
        self.node_features = self.__init_node_features(node_features)
        self.edge_features = self.__init_edge_features(edge_features)
        self.edge_weights = self.__init_edge_weights(edge_weights)
        self.graph_features = graph_features
        self._nx_repr = None
                
    def get_nx(self):
        if not self._nx_repr:
            self._nx_repr = self._build_nx()
        return deepcopy(self._nx_repr)
    
    def __init_node_features(self, node_features):
        return np.zeros((self.data.shape[0], 1)) if isinstance(node_features, (str, type(None))) else node_features

    def __init_edge_features(self, edge_features):
        edges = np.nonzero(self.data)
        return np.zeros((len(edges[0]), 1)) if isinstance(edge_features, (str, type(None))) else edge_features
    
    def __init_edge_weights(self, edge_weights):
        edges = np.nonzero(self.data)
        return np.zeros((len(edges[0]), 1)) if isinstance(edge_weights, (str, type(None))) else edge_weights
    
    def _build_nx(self):
        nx_repr = nx.from_numpy_array(self.data)
        nx_repr.add_nodes_from([node, {'node_features': self.node_features[node]}] for node in nx_repr.nodes())
        edges = list(nx_repr.edges)
        nx_repr.add_edges_from([(edge[0], edge[1], {'edge_features': self.edge_features[i]}) for i, edge in enumerate(edges)])
        return nx_repr
    
    @property
    def num_edges(self):
        nx_repr = self.get_nx()
        return nx_repr.number_of_edges()
            
    
    
        