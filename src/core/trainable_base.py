from abc import ABCMeta, abstractmethod
from src.core.savable import Savable
from src.utils.context import Context

class Trainable(Savable,metaclass=ABCMeta):
    
    def __init__(self, context: Context, local_config) -> None:
        super().__init__(context,local_config)        
        ##############################################################################
        # fit the model on a specific dataset
        # or read if already existing
        self.dataset = self.local_config['dataset']
        self.local_config['parameters']['fold_id'] =  self.local_config['parameters'].get('fold_id', -1)
        #TODO: Add getDefault Method that return the default conf snippet of parameters conf node.
        self.local_config = self.check_configuration(self.local_config)
        # real init details
        self.init()
        # retrain if explicitely specified or if the weights of the model don't exists
        if self._to_retrain() or not self.saved():
            self.context.logger.info("Need to be train: "+str(self))
            self.fit()
            self.context.logger.info("Trained: "+str(self))
        else:
            self.read()
            self.context.logger.info("Loaded: "+str(self))
        ##############################################################################

    def _to_retrain(self):
        retrain = self.local_config['parameters'].get('retrain', False)
        self.local_config['parameters']['retrain']= False
        return retrain

    def fit(self):
        self.real_fit()
        self.write()
        self.context.logger.info(str(self)+" saved.")
    
    @abstractmethod
    def init(self):
        pass
    
    @abstractmethod
    def check_configuration(self, local_config):
        pass

    @abstractmethod
    def real_fit(self):
        pass
    