import random
import itertools
import numpy as np

from src.dataset.data_instance_base import DataInstance
from src.dataset.dataset_base import Dataset
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle


class PerturbationRandExplainer(Explainer):
    
    def __init__(self,
                id,
                perturbation_percentage=.05,
                fold_id=0,
                config_dict=None) -> None:
        
        super().__init__(id, config_dict)
        
        self.perturbation_percentage = perturbation_percentage
        self.fold_id = fold_id
        
        
    def explain(self, instance, oracle: Oracle, dataset: Dataset):
        self.name = f'perturbation_rand-dataset_{dataset.name}-fold_id={self.fold_id}'
        print(dataset.get_split_indices()[self.fold_id]['test'])
        adj_matrix = instance.to_numpy_array()
        nodes = adj_matrix.shape[0]
        # all edges (direct graph)
        all_edges = list(itertools.product(list(range(nodes)), repeat=2))
        # filter for only undirected edges
        new_edges = list()
        for edge in all_edges:
            if ((edge[1], edge[0]) not in new_edges) and edge[0] != edge[1]:
                new_edges.append(list(edge))
        new_edges = np.array(new_edges)
        # sample according to perturbation_percentage
        
        sample_index = np.random.choice(list(range(len(new_edges))),
                                         size=int(len(new_edges) * self.perturbation_percentage))
        
        sampled_edges = new_edges[sample_index]
        # switch on/off the sampled edges
        adj_matrix[sampled_edges[:,0], sampled_edges[:,1]] = 1 - adj_matrix[sampled_edges[:,0], sampled_edges[:,1]]
        adj_matrix[sampled_edges[:,1], sampled_edges[:,0]] = 1 - adj_matrix[sampled_edges[:,1], sampled_edges[:,0]]
    
        # build the instance
        cf_instance = DataInstance(instance.id)
        cf_instance.from_numpy_array(adj_matrix)
        return cf_instance
    