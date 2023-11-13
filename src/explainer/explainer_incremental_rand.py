import random
import itertools
import numpy as np

from src.dataset.data_instance_base import DataInstance
from src.dataset.dataset_base import Dataset
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle


class IRandExplainer(Explainer):
    
    def __init__(self,
                id,
                perturbation_percentage=.05,
                tries=1,
                fold_id=0,
                config_dict=None) -> None:
        
        super().__init__(id, config_dict)
        
        self.perturbation_percentage = perturbation_percentage
        self.tries = tries
        self.fold_id = fold_id
        self.name = 'i_rand'
        
        
    def explain(self, instance, oracle: Oracle, dataset: Dataset):
        l_input_inst = oracle.predict(instance)

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
        
        # Calculate the maximun percent of edges to modify
        k = [int(len(new_edges) * i%100) for i in range(1, int(self.perturbation_percentage*100))]

        # increase the number of random modifications
        for i in k:
            # how many attempts at a current modification level
            for j in range(0, self.tries):
                cf_cand_matrix = np.copy(adj_matrix)
                # sample according to perturbation_percentage
                sample_index = np.random.choice(list(range(len(new_edges))), size=i)
                sampled_edges = new_edges[sample_index]

                # switch on/off the sampled edges
                cf_cand_matrix[sampled_edges[:,0], sampled_edges[:,1]] = 1 - cf_cand_matrix[sampled_edges[:,0], sampled_edges[:,1]]
                cf_cand_matrix[sampled_edges[:,1], sampled_edges[:,0]] = 1 - cf_cand_matrix[sampled_edges[:,1], sampled_edges[:,0]]
            
                # build the counterfactaul candidates instance
                cf_cand_instance = DataInstance(instance.id)
                cf_cand_instance.from_numpy_array(cf_cand_matrix)

                # if a counterfactual was found return that
                l_cf_cand = oracle.predict(cf_cand_instance)
                if l_input_inst != l_cf_cand:
                    return cf_cand_instance
        
        # If no counterfactual was found return the original instance by convention
        return instance
    