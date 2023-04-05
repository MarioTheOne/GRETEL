import numpy as np

from src.dataset.converters.abstract_converter import ConverterAB
from src.dataset.data_instance_base import DataInstance
from src.dataset.data_instance_features import DataInstanceWFeaturesAndWeights


class DefaultFeatureAndWeightConverter(ConverterAB):
    
    def __init__(self, feature_dim=10, weight_dim=1):
        super(DefaultFeatureAndWeightConverter, self).__init__()
        
        self.name = 'weight_converter'
        
        self.feature_dim = feature_dim
        self.weight_dim = weight_dim
        
    def convert_instance(self, instance: DataInstance) -> DataInstanceWFeaturesAndWeights:
        converted_instance = DataInstanceWFeaturesAndWeights(instance.id,
                                                graph=instance.graph,
                                                graph_label=instance.graph_label,
                                                node_labels=instance.node_labels,
                                                edge_labels=instance.edge_labels,
                                                graph_dgl=instance.graph_dgl)

        if not hasattr(instance, 'features'):
            converted_instance.features = self.__create_dummy(instance, self.feature_dim)
        
        if not hasattr(instance, 'weights'):
            converted_instance.weights = self.__create_dummy(instance, self.weight_dim)
        
        return converted_instance
    
    def __create_dummy(self, instance: DataInstance, dim: int):
        return np.random.normal(0, 1, (instance.graph.number_of_nodes(), dim))