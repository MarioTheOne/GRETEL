from types import SimpleNamespace
from typing import List

import numpy as np
from src.n_dataset.instances.graph import GraphInstance
import torch

from src.dataset.data_instance_base import DataInstance
from src.dataset.data_instance_features import DataInstanceWFeaturesAndWeights
from src.core.oracle_base import Oracle
from src.utils.samplers.abstract_sampler import Sampler


class PositiveAndNegativeEdgeSampler(Sampler):
    
    def __init__(self, sampling_iterations):
        super().__init__()
        self._name = 'positive_and_negative_edge_sampler'
        self.sampling_iterations = sampling_iterations      
          
    def sample(self, instance: DataInstance, oracle: Oracle, **kwargs) -> List[DataInstance]:
        kwargs = SimpleNamespace(**kwargs)
        
        edge_probs = kwargs.edge_probabilities
        node_features = kwargs.node_features
        edge_list = kwargs.edge_index
        edge_num = instance.num_edges
        
        cf_instance = self.__sample(instance, node_features, edge_probs[edge_list[0,:], edge_list[1,:]], edge_list, num_samples=edge_num) 
        if oracle.predict(cf_instance) != oracle.predict(instance):
            return cf_instance
        else:
            # get the "negative" edges that aren't estimated
            missing_edges = self.__negative_edges(edge_list, instance.num_nodes).T
            edge_probs = torch.from_numpy(np.array([1 / len(missing_edges) for _ in range(len(missing_edges))]))
            # check sampling for sampling_iterations
            # and see if we find a valid counterfactual
            for _ in range(self.sampling_iterations):
                cf_instance = self.__sample(cf_instance, node_features, edge_probs, missing_edges)
                if oracle.predict(cf_instance) != oracle.predict(instance):
                    return cf_instance
        return None 
                
    def __negative_edges(self, edges, num_vertices):
        i, j = np.triu_indices(num_vertices, k=1)
        all_edges = set(list(zip(i, j)))
        edges = set([tuple(x) for x in edges])
        return torch.from_numpy(np.array(list(all_edges.difference(edges))))
  
    def __sample(self, instance, features, probabilities, edge_list, num_samples=1):
        n_nodes = instance.num_nodes
        adj = torch.zeros((n_nodes, n_nodes)).double()
        selected_indices = torch.multinomial(probabilities, num_samples=num_samples, replacement=True).numpy()
        adj[edge_list[:,selected_indices], edge_list[:,selected_indices]] = 1
        return GraphInstance(id=instance.id, label="dummy", data=adj.numpy(), node_features=features.numpy())        
