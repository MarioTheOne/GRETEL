
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from src.oracle.tabulars.tabular_base import TabularOracle
from src.utils.cfg_utils import init_dflts_to_of

class SVMOracle(TabularOracle):

    def init(self):        
        svm = LinearSVC(**self.local_config['parameters']['model']['parameters'])
        self.model = CalibratedClassifierCV(svm)
        super().init()

    def check_configuration(self):
        super().check_configuration()
        kls="sklearn.svm.LinearSVC"
        init_dflts_to_of(self.local_config, 'model', kls) #Init the default accordingly to the nested Classifier
