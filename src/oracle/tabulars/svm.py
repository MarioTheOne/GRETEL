
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from src.oracle.tabulars.tabular_base import TabularOracle

class SVMOracle(TabularOracle):

    def init(self):
        super().init()
        svm = LinearSVC(**self.local_config['parameters']['model']['parameters'])
        self.model = CalibratedClassifierCV(svm) 
