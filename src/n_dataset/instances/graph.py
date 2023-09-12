from copy import deepcopy

import networkx as nx
import numpy as np

from src.n_dataset.instances.base import DataInstance

class GraphInstance(DataInstance):

    def __init__(self, id, label, data, node_features=None, edge_features=None, graph_features=None, dataset=None):
        super().__init__(id, label, data, dataset=dataset)
        self.node_features = self.__init_node_features(node_features)
        self.edge_features = self.__init_edge_features(edge_features)
        self.graph_features = graph_features
        self._nx_repr = None
                
    def get_nx(self):
        if not self._nx_repr:
            self._nx_repr = self._build_nx()
        return deepcopy(self._nx_repr)
    
    def __init_node_features(self, node_features):
        return np.zeros((self.data.shape[0], 1)) if not node_features else node_features

    def __init_edge_features(self, edge_features):
        edges = np.nonzero(self.data)
        return np.zeros((len(edges[0]), 1)) if not edge_features else edge_features
    
    def _build_nx(self):
        nx_repr = nx.from_numpy_array(self.data)
        nx_repr.add_nodes_from([node, {'node_features': self.node_features[node]}] for node in nx_repr.nodes())
        edges = list(nx_repr.edges)
        nx_repr.add_edges_from([(edge[0], edge[1], {'edge_features': self.edge_features[i]}) for i, edge in enumerate(edges)])
        return nx_repr
            
    
    
        