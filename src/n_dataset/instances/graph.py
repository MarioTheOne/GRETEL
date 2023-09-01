import networkx as nx

from src.dataset.instances.base import DataInstance
from copy import deepcopy

class GraphInstance(DataInstance):

    def __init__(self, id, label, data, features=None, weights=None, dataset=None):
        super().__init__(id, label, data, dataset=dataset)
        self.features = features
        self.weights = weights
        # graph measure dictionary
        self._graph_measures = {}
        self._nx_repr = self.__build_nx()
        
    def get_nx(self):
        return deepcopy(self._nx_repr)
    
    def __build_nx(self):
        nx_repr = nx.from_numpy_array(self.data)
        nx_repr.add_nodes_from([node, {'features': self.features[node]}] for node in nx_repr.nodes())
        edges = list(nx_repr.edges)
        nx_repr.add_edges_from([edge[0], edge[1], {'weights': self.weights[edge[0], edge[1]]} for edge in edges])
        return nx_repr
            
    
    
        