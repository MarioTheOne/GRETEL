
from sklearn.svm import LinearSVC

from src.oracle.tabulars.predictor import TabularOracle


class SVMOracle(TabularOracle):

    def init(self):
        super().init()
        self.model = LinearSVC(**self.local_config['parameters']['model']['parameters'])