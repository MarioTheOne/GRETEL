import numpy as np
from src.dataset.converters.weights_converter import \
    DefaultFeatureAndWeightConverter
from src.dataset.data_instance_base import DataInstance
from src.dataset.data_instance_features import DataInstanceWFeaturesAndWeights

"""
    Adaptation of Tree-Cycles conversion based on the
    graph classification converter from
    the original repository
"""
class CF2TreeCycleConverter(DefaultFeatureAndWeightConverter):
    
    def __init__(self, feature_dim=10):
        super(CF2TreeCycleConverter, self).__init__()
        self.name = 'cf2_converter'
        self.feature_dim = feature_dim
        
    def convert_instance(self, instance: DataInstance) -> DataInstanceWFeaturesAndWeights:
        converted_instance = super().convert_instance(instance)
        weights, features, adj_matrix = self.__preprocess(converted_instance)
        converted_instance.weights = weights
        converted_instance.features = features
        converted_instance.from_numpy_array(adj_matrix)
        return converted_instance
        
    def __preprocess(self, instance: DataInstanceWFeaturesAndWeights) -> np.ndarray:
        adj = instance.to_numpy_array()
        n_nodes = len(adj)
        # one-hot feature encoding
        features = np.eye(n_nodes)[np.random.choice(self.feature_dim, n_nodes)]
        # define the new adjacency matrix which is a full one matrix
        new_adj = np.where(adj != 0, 1, 0)
        # the weights need to be an array of real numbers with
        # length equal to the number of edges
        row_indices, col_indices = np.where(adj != 0)
        weights = adj[row_indices, col_indices]
      
        return weights, features, new_adj