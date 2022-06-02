from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.dataset.data_instance_base import DataInstance
from src.oracle.oracle_base import Oracle

class GraphEditDistanceMetric(EvaluationMetric):
    """Provides a graph edit distance function for graphs where nodes are already matched, 
    thus eliminating the need of performing an NP-Complete graph matching.
    """

    def __init__(self, node_insertion_cost=1.0, node_deletion_cost=1.0, edge_insertion_cost=1.0,
                 edge_deletion_cost=1.0, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Graph_Edit_Distance'
        self._node_insertion_cost = node_insertion_cost
        self._node_deletion_cost = node_deletion_cost
        self._edge_insertion_cost = edge_insertion_cost
        self._edge_deletion_cost = edge_deletion_cost
        

    def evaluate(self, instance_1: DataInstance, instance_2: DataInstance, oracle: Oracle = None):
        G1 = instance_1.graph
        G2 = instance_2.graph

        edit_distance = 0.0

        for n in G1.nodes:
            if not G2.has_node(n):
                edit_distance += self._node_deletion_cost

        for n in G2.nodes:
            if not G1.has_node(n):
                edit_distance += self._node_insertion_cost

        for e in G1.edges:
            if not G2.has_edge(*e):
                edit_distance += self._edge_deletion_cost

        for e in G2.edges:
            if not G1.has_edge(*e):
                edit_distance += self._edge_insertion_cost

        return edit_distance
    