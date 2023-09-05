import networkx as nx

from src.n_dataset.instances.base import DataInstance
from copy import deepcopy

class GraphInstance(DataInstance):

    def __init__(self, id, label, data, node_features=None, edge_features=None, graph_features=None, dataset=None):
        super().__init__(id, label, data, dataset=dataset)
        self.node_features = node_features
        self.edge_features = edge_features
        self.graph_features = graph_features
        self._nx_repr = self.__build_nx()
        
    def get_nx(self):
        return deepcopy(self._nx_repr)
    
    def __build_nx(self):
        nx_repr = nx.from_numpy_array(self.data)
        if self.node_features:
            nx_repr.add_nodes_from([node, {'node_features': self.node_features[node]}] for node in nx_repr.nodes())
        if self.edge_features:
            edges = list(nx_repr.edges)
            nx_repr.add_edges_from([(edge[0], edge[1], {'edge_features': self.edge_features[edge[0], edge[1]]}) for edge in edges])
        return nx_repr
            
    
    
        