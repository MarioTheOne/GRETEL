
from sklearn.svm import LinearSVC

from src.oracle.tabulars.tabular_base import TabularOracle


class SVMOracle(TabularOracle):

    def init(self):
        super().init()
        self.model = LinearSVC(**self.local_config['parameters']['model']['parameters'])