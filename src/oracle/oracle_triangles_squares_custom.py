from src.dataset.dataset_base import Dataset
from src.oracle.oracle_base import Oracle

import numpy as np

class TrianglesSquaresCustomOracle(Oracle):

    def __init__(self, id, oracle_store_path, config_dict=None) -> None:
        super().__init__(id, oracle_store_path, config_dict)
        self._name = 'triangles_squares_custom_oracle'

    def fit(self, dataset: Dataset, split_i=-1):
        pass

    def _real_predict(self, data_instance):
        # Classify
        if len(data_instance.graph.nodes) == 3 and len(data_instance.graph.edges) == 3:
            return 1 # triangle
        else:
            return 0 # other shape (squares)

    def embedd(self, instance):
        return instance

    def write_oracle(self):
        pass

    def read_oracle(self, oracle_name):
        pass
