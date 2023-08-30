from sklearn.neighbors import KNeighborsClassifier
from src.oracle.tabulars.predictor import TabularOracle

class KNNOracle(TabularOracle):

    def init(self):
        super().init()
        self.model = KNeighborsClassifier(**self.local_config['parameters']['model']['parameters'])         

    def check_configuration(self, local_config):
        params = local_config['parameters']['model']['parameters']
        if "n_neighbors" not in params:
            params['n_neighbors'] = 3
        return super().check_configuration(local_config)