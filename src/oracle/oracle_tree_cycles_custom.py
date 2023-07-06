from src.dataset.dataset_base import Dataset
from src.oracle.oracle_base import Oracle

import numpy as np
import networkx as nx

class TreeCyclesCustomOracle(Oracle):

    def __init__(self, id, oracle_store_path, config_dict=None) -> None:
        super().__init__(id, oracle_store_path, config_dict)
        self._name = 'tree_cycles_custom_oracle'

    def fit(self, dataset: Dataset, split_i=-1):
        pass

    def _real_predict(self, data_instance):
        try:
            nx.find_cycle(data_instance.graph, orientation='ignore')
            return 1
        except nx.exception.NetworkXNoCycle:
            return 0
        
    def _real_predict_proba(self, data_instance):
        # softmax-style probability predictions
        try:
            nx.find_cycle(data_instance.graph, orientation='ignore')
            return np.array([[0,1]])
        except nx.exception.NetworkXNoCycle:
            return np.array([[1,0]])

    def embedd(self, instance):
        return instance

    def write_oracle(self):
        pass

    def read_oracle(self, oracle_name):
        pass