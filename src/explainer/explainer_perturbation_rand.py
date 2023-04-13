import random

from src.dataset.data_instance_base import DataInstance
from src.dataset.dataset_base import Dataset
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle


class PerturbationRandExplainer(Explainer):
    
    def __init__(self,
                id,
                perturbation_cycles=10,
                fold_id=0,
                config_dict=None) -> None:
        
        super().__init__(id, config_dict)
        
        self.perturbation_cycles = perturbation_cycles
        self.fold_id = fold_id
        
        
    def explain(self, instance, oracle: Oracle, dataset: Dataset):
        self.name = f'perturbation_rand-dataset_{dataset.name}-fold_id={self.fold_id}'
        print(dataset.get_split_indices()[self.fold_id]['test'])
        adj_matrix = instance.to_numpy_array()
        nodes = adj_matrix.shape[0]
        
        for _ in range(self.perturbation_cycles):
            r = list(range(nodes))
            u = random.choice(r)
            r = list(range(u)) + list(range(u+1, nodes))
            v = random.choice(r)
            
            adj_matrix[u][v] = 1 - adj_matrix[u][v]
            
        
        cf_instance = DataInstance(instance.id)
        cf_instance.from_numpy_array(adj_matrix)
        return cf_instance
    