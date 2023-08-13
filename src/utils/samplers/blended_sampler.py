from types import SimpleNamespace
from typing import List

import numpy as np
import torch

from src.dataset.data_instance_base import DataInstance
from src.dataset.data_instance_features import DataInstanceWFeaturesAndWeights
from src.oracle.oracle_base import Oracle
from src.utils.samplers.abstract_sampler import Sampler


class BlendedSampler(Sampler):
    
    def __init__(self, sampling_iterations):
        super().__init__()
        self._name = 'blended_sampler'
        self.sampling_iterations = sampling_iterations      
          
    def sample(self, instance: DataInstance, oracle: Oracle, **kwargs) -> List[DataInstance]:
        kwargs = SimpleNamespace(**kwargs)
        
        edge_probs = kwargs.edge_probabilities
        embedded_features = kwargs.embedded_features
        edge_list = kwargs.edge_index
        
        # Summing all tensors in the dictionary
        sum_tensor = torch.zeros_like(next(iter(edge_probs.values())))  # Initialize the sum tensor with zeros
        for tensor in edge_probs.values():
            sum_tensor += tensor
        sum_tensor /= len(edge_probs.values())
        # average the features
        average_features = torch.zeros_like(next(iter(embedded_features.values())))
        for tensor in embedded_features.values():
            average_features += tensor
        average_features /= len(embedded_features.values())

        edge_num = instance.graph.number_of_edges()
        """for _ in range(self.sampling_iterations):
            cf_instance = self.__sample(instance, average_features, sum_tensor, edge_list, num_samples=edge_num) 
            if oracle.predict(cf_instance) != oracle.predict(instance):
                return cf_instance
        return None """
    
        cf_instance = self.__sample(instance, average_features, sum_tensor[edge_list], edge_list, num_samples=edge_num) 
        if oracle.predict(cf_instance) != oracle.predict(instance):
            return cf_instance
        else:
            # get the "negative" edges
            missing_edges = self.__negative_edges(edge_list, instance.graph.number_of_nodes())
            edge_probs = torch.from_numpy(np.array([1 / len(missing_edges) for _ in range(len(missing_edges))]))

            #edge_probs = sum_tensor[missing_edges]
            # check sampling for sampling_iterations
            # and see if we find a valid counterfactual
            for _ in range(self.sampling_iterations):
                cf_instance = self.__sample(cf_instance, average_features, edge_probs, missing_edges, num_samples=2)
                if oracle.predict(cf_instance) != oracle.predict(instance):
                    return cf_instance
        return None 
  
    def __negative_edges(self, edges, num_vertices):
        i, j = np.triu_indices(num_vertices, k=1)
        all_edges = set(list(zip(i, j)))
        edges = set([tuple(x) for x in edges])
        return torch.from_numpy(np.array(list(all_edges.difference(edges))))
    
    def __sample(self, instance: DataInstance, features, probabilities, edge_list, num_samples=1):
        """n_nodes = instance.graph.number_of_nodes()
        adj = torch.zeros((n_nodes, n_nodes)).double()
        weights = torch.zeros((n_nodes, n_nodes)).double()
        
        print(edge_list)
        print(probabilities.shape)
        
        upper_triangle_indices = torch.triu_indices(probabilities.size(0), probabilities.size(1), offset=1)
        # Index the tensor using the upper triangle indices
        upper_triangle_values = probabilities[upper_triangle_indices[0], upper_triangle_indices[1]]
        # Flatten the upper triangle values
        flattened_upper_triangle = upper_triangle_values.flatten()
        ##################################################
        cf_instance = DataInstanceWFeaturesAndWeights(id=instance.id)
        try:
            selected_indices = torch.multinomial(flattened_upper_triangle, num_samples=num_samples, replacement=True).numpy()
            print(selected_indices)
            for index in selected_indices:
                adj[index // n_nodes, index % n_nodes] = 1
            adj = adj + adj.T - torch.diag(torch.diag(adj))

            "weights[selected_indices] = probabilities[selected_indices]
            weights = weights + weights.T - torch.diag(torch.diag(weights))
            
            cf_instance.from_numpy_array(adj.numpy())
            #cf_instance.weights = weights.numpy()
            cf_instance.features = features.numpy()
        except RuntimeError: # the probabilities are all zero
            cf_instance.from_numpy_array(adj.numpy())
        
        return cf_instance"""
        
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