from abc import ABC
import networkx as nx
import numpy as np


class DataInstance(ABC):

    def __init__(self,
                 id=None,
                 name: str = None,
                 graph=None,
                 graph_dgl=None,
                 graph_label: int = None,
                 node_labels: dict = None,
                 edge_labels: dict = None,
                 mcd: int = None) -> None:
        self._id = id
        self._name = name
        self._graph = graph
        self._graph_dgl = graph_dgl
        self._graph_label = graph_label
        self._node_labels = node_labels
        self._edge_labels = edge_labels
        self._mcd = mcd
        self._np_array = None
        self._n_node_types = 0
        self._max_n_nodes = 0
        super().__init__()

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, new_id):
        self._id = new_id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    @property
    def graph(self):
        return self._graph
   
    @graph.setter
    def graph(self, new_graph):     
        self._graph = new_graph

    @property
    def graph_dgl(self):
        return self._graph_dgl

    @graph_dgl.setter
    def graph_dgl(self, new_graph):
        self._graph_dgl = new_graph

    @property
    def graph_label(self) -> int:
        return self._graph_label

    @graph_label.setter
    def graph_label(self, new_label: int):
        self._graph_label = new_label

    @property
    def node_labels(self) -> dict:
        return self._node_labels

    @node_labels.setter
    def node_labels(self, new_labels: dict):
        self._node_labels = new_labels

    @property
    def edge_labels(self) -> dict:
        return self._edge_labels

    @edge_labels.setter
    def edge_labels(self, new_labels: dict):
        self._edge_labels = new_labels

    @property
    def minimum_counterfactual_distance(self):
        return self._mcd

    @minimum_counterfactual_distance.setter
    def minimum_counterfactual_distance(self, new_value: int):
        self._mcd = new_value

    @property
    def max_n_nodes(self):
        return self._max_n_nodes

    @max_n_nodes.setter
    def max_n_nodes(self, new_val):
        self._max_n_nodes = new_val

    @property
    def n_node_types(self):
        return self._n_node_types

    @n_node_types.setter
    def n_node_types(self, new_val):
        self._n_node_types = new_val

    def to_numpy_matrix(self):
        return nx.to_numpy_matrix(self.graph)

    def from_numpy_matrix(self, np_adj_matrix):
        self.graph = nx.from_numpy_matrix(np_adj_matrix)

    def to_numpy_array(self, store=True):

        # If the instance does not contain a numpy array
        if (self._np_array is None):
            # We transform the adjacency matrix to a 32bits integer for being consistent with the reading of some datasets
            result = nx.to_numpy_array(self.graph, dtype=np.int32)

            # Store the numpy array in the instance
            if (store):
                self._np_array = result

            # return the numpy array
            return result
        else:
            # Return the stored numpy array
            return self._np_array
        
    def node_degrees(self):
        return [val for _, val in self.graph.degree()]

    def from_numpy_array(self, np_adj_matrix, store=False):
        # If store is true we should store the numpy array adjacency matrix
        if store:
            self._np_array = np_adj_matrix

        self.graph = nx.from_numpy_array(np_adj_matrix)


    def to_numpy_arrays(self, store=False, max_n_nodes=-1, n_node_types=-1):
        """Argument for the RD2NX function should be a valid SMILES sequence
        returns: the graph
        """
        n_nodes = max_n_nodes
        n_ntypes = n_node_types

        if max_n_nodes == -1 or n_node_types == -1:
            n_nodes = self.max_n_nodes
            n_ntypes = self.n_node_types

        nodes = np.zeros((n_nodes, n_ntypes))

        # All nodes are of the same type in the general case
        for i in self.graph.nodes:
            nodes[i, 0] = 1

        adj = np.zeros((n_nodes, n_nodes))

        for u, v in self.graph.edges:
            adj[u, v] = 1
            adj[v, u] = 1

        adj += np.eye(n_nodes)

        return nodes, adj