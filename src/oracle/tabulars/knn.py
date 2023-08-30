from src.dataset.dataset_base import Dataset
from src.core.embedder_base import Embedder
from src.oracle.oracle_base import Oracle

from sklearn.neighbors import KNeighborsClassifier
import os
import joblib
from src.oracle.tabular_oracles.predictor import TabularOracle
from src.utils.utils import get_only_default_params

class KNNOracle(TabularOracle):

    def init(self):
        super().init()
        self.model = KNeighborsClassifier(**self.local_config['parameters']['model']['parameters'])         

    def check_configuration(self, local_config):
        params = local_config['parameters']['model']['parameters']
        if "n_neighbors" not in params:
            params['n_neighbors'] = 3
            
        return super().check_configuration(params)