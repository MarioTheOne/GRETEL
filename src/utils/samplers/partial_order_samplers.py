from types import SimpleNamespace
from typing import List

import numpy as np
import torch

from src.dataset.data_instance_base import DataInstance
from src.dataset.data_instance_features import DataInstanceWFeaturesAndWeights
from src.oracle.oracle_base import Oracle
from src.utils.samplers.abstract_sampler import Sampler


class PositiveAndNegativeEdgeSampler(Sampler):
    
    def __init__(self, sampling_iterations):
        super().__init__()
        self._name = 'positive_and_negative_edge_sampler'
        self.sampling_iterations = sampling_iterations      
          
    def sample(self, instance: DataInstance, oracle: Oracle, **kwargs) -> List[DataInstance]:
        kwargs = SimpleNamespace(**kwargs)
        
        edge_probs = kwargs.edge_probabilities
        embedded_features = kwargs.embedded_features
        edge_list = kwargs.edge_index
        
        pred_label = oracle.predict(instance)
        
        edge_probs = edge_probs.get(pred_label)
        embedded_features = embedded_features.get(pred_label)
                
        edge_num = instance.graph.number_of_edges()
        cf_instance = self.__sample(instance, embedded_features, edge_probs[edge_list], edge_list, num_samples=edge_num) 
        if oracle.predict(cf_instance) != oracle.predict(instance):
            return cf_instance
        else:
            # get the "negative" edges that aren't estimated
            missing_edges = self.__negative_edges(edge_list, instance.graph.number_of_nodes())
            edge_probs = torch.from_numpy(np.array([1 / len(missing_edges) for _ in range(len(missing_edges))]))
            # check sampling for sampling_iterations
            # and see if we find a valid counterfactual
            for _ in range(self.sampling_iterations):
                cf_instance = self.__sample(cf_instance, embedded_features, edge_probs, missing_edges)
                if oracle.predict(cf_instance) != oracle.predict(instance):
                    return cf_instance
        return None 
                
    def __negative_edges(self, edges, num_vertices):
        i, j = np.triu_indices(num_vertices, k=1)
        all_edges = set(list(zip(i, j)))
        edges = set([tuple(x) for x in edges])
        return torch.from_numpy(np.array(list(all_edges.difference(edges))))
  
    def __sample(self, instance: DataInstance, features, probabilities, edge_list, num_samples=1):
        n_nodes = instance.graph.number_of_nodes()
        adj = torch.zeros((n_nodes, n_nodes)).double()
        weights = torch.zeros((n_nodes, n_nodes)).double()
        ##################################################
        cf_instance = DataInstanceWFeaturesAndWeights(id=instance.id)
        try:
            selected_indices = torch.multinomial(probabilities, num_samples=num_samples, replacement=True).numpy()
            adj[edge_list[selected_indices]] = 1
            adj = adj + adj.T - torch.diag(torch.diag(adj))
            
            weights[edge_list[selected_indices]] = probabilities[selected_indices]
            weights = weights + weights.T - torch.diag(torch.diag(weights))
            
            cf_instance.from_numpy_array(adj.numpy())
            cf_instance.weights = weights.numpy()
            cf_instance.features = features.numpy()
        except RuntimeError: # the probabilities are all zero
            cf_instance.from_numpy_array(adj.numpy())
        
        return cf_instance