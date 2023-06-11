import numpy as np

from src.dataset.data_instance_base import DataInstance
from src.dataset.dataset_base import Dataset
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle
from src.evaluation.actionability_evaluator_base import ActionabilityEvaluator


class BLActionabilityEvaluator(ActionabilityEvaluator):

    def __init__(self, id, 
                 restricted_nodes=[], 
                 restricted_edges=[], 
                 restricted_node_features=[], 
                 restricted_edge_features=[], 
                 config_dict=None):
        # Initializing the base class
        super().__init__(id=id, config_dict=config_dict)
        # Initializing the BLActionabilityExplainer
        self.name = 'abstract_actionability_evaluator'
        self._restricted_nodes = restricted_nodes
        self._restricted_edges = restricted_edges
        self._restricted_node_features = restricted_node_features
        self._restricted_edge_features = restricted_edge_features


    def evaluate_graph(self, 
                       original_instance : DataInstance, 
                       cf_instance : DataInstance):
        
        # Implementation for numpy matrices
        A_original = original_instance.to_numpy_array()
        A_cf = cf_instance.to_numpy_array()

        # Get the difference in the number of nodes
        nodes_diff_count = abs(A_original.shape[0] - A_cf.shape[0])
        min_num_nodes = min(A_original.shape[0], A_cf.shape[0])

        # Check if any non actionable node was changed
        for i in range(0, nodes_diff_count):
            if (min_num_nodes + i - 1) in self._restricted_nodes:
                return False 

        # Get the shape of the matrices
        shape_A_original = A_original.shape
        shape_A_cf = A_cf.shape

        # Find the minimum dimensions of the matrices
        min_shape = (min(shape_A_original[0], shape_A_cf[0]), min(shape_A_original[1], shape_A_cf[1]))

        # Initialize an empty list to store the differences
        edges_diff = []

        # Iterate over the common elements of the matrices and check if those edges were modified
        for i in range(min_shape[0]):
            for j in range(min_shape[1]):
                if A_original[i,j] != A_cf[i,j] and (i, j) in self._restricted_edges:
                    return False

        # If the matrices have different shapes
        if shape_A_original != shape_A_cf:
            max_shape = np.maximum(shape_A_original, shape_A_cf)

            # Loop through the remaining cells in the larger matrix (the matrixes are square shaped)
            for i in range(min_shape[0], max_shape[0]):
                for j in range(min_shape[1], max_shape[1]):
                    if shape_A_original > shape_A_cf:
                        # If the original have more nodes than the cf and there are edges towards them
                        if A_original[i,j] and (i,j) in self._restricted_edges:
                            return False
                    else:
                        # If the CF have more nodes than the original and there are edges towards them
                        if A_cf[i,j] and (i,j) in self._restricted_edges:
                            return False
                        
        # No restricted nodes or edges where modified
        return True


    def evaluate_edge(self,
                      original_instance : DataInstance, 
                      cf_instance : DataInstance, 
                      target_edge):
        # Check if the target edge is in the restricted list
        if target_edge in self._restricted_edges:
            return False
        
        return True


    def evaluate_node(self,
                      original_instance : DataInstance, 
                      cf_instance : DataInstance, 
                      target_node):
        # Check if the target node is in the restricted list
        if target_node in self._restricted_nodes:
            return False
        
        return True


    def evaluate_node_features(self,
                               original_instance : DataInstance, 
                               original_features, 
                               new_features):
        # Check if any of the modified node features are in the restricted list
        for i in range(0, len(original_features)):
            if new_features[i] != original_features[i] and i in self._restricted_node_features:
                return False
            
        return True
    

    def evaluate_edge_features(self,
                               original_instance : DataInstance, 
                               original_features, 
                               new_features):
        # Check if any of the modified edge features are in the restricted list
        for i in range(0, len(original_features)):
            if new_features[i] != original_features[i] and i in self._restricted_edge_features:
                return False
            
        return True