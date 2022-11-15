import networkx as nx
import torch

from src.dataset.data_instance_target_node import TargetNodeDataInstance


class SynDataInstance(TargetNodeDataInstance):
    """

    """
    def __init__(self,
                 id = None,
                 name: str=None, 
                 graph_data=None,
                 graph_label: int=None, 
                 node_labels: dict=None, 
                 edge_labels: dict=None,
                 target_node: int=None,
                 mcd: int=None) -> None:
        super().__init__(id, name, graph_label, node_labels, edge_labels, mcd, target_node = target_node)
        self._graph_data = graph_data

    @property
    def max_n_nodes(self):
        return self._max_mol_len

    @max_n_nodes.setter
    def max_n_nodes(self, new_val):
        self._max_mol_len = new_val

    @property
    def n_node_types(self):
        return self._max_n_atoms

    @n_node_types.setter
    def n_node_types(self, new_val):
        self._max_n_atoms = new_val

    @property
    def graph_data(self):
        return self._graph_data

    @graph_data.setter
    def graph_data(self, new_val):
        self._graph_data = new_val
