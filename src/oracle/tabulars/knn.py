from sklearn.neighbors import KNeighborsClassifier
from src.oracle.tabulars.tabular_base import TabularOracle
from src.utils.cfg_utils import init_dflts_to_of

class KNNOracle(TabularOracle):

    def init(self):

        self.model = KNeighborsClassifier(**self.local_config['parameters']['model']['parameters'])
        super().init()

    def check_configuration(self):
        super().check_configuration()
        kls="sklearn.neighbors.KNeighborsClassifier"
        init_dflts_to_of(self.local_config, 'model', kls, n_neighbors=3) #Init the default accordingly to the nested Classifier