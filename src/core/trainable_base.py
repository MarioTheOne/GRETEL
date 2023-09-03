import pickle
from flufl.lock import Lock
from datetime import timedelta
from abc import ABCMeta, abstractmethod
from src.core.savable import Savable
from src.utils.context import Context

class Trainable(Savable,metaclass=ABCMeta):
    
    def __init__(this, context: Context, local_config) -> None:
        super().__init__(context,local_config)        
        ##############################################################################
        # fit the model on a specific dataset
        # or read if already existing
        this.dataset = this.local_config['dataset']
        this.local_config['parameters']['fold_id'] =  this.local_config['parameters'].get('fold_id', -1)
        #TODO: Add getDefault Method that return the default conf snippet of parameters conf node.
        this.local_config = this.check_configuration(this.local_config)
        # real init details
        this.init()
        # retrain if explicitely specified or if the weights of the model don't exists

        lock = Lock(this.context.get_path(this)+'.lck',lifetime=timedelta(hours=this.context.lock_release_tout))
        with lock:
            if this._to_retrain() or not this.saved():
                this.context.logger.info("Need to be train: "+str(this))
                this.fit()
                this.context.logger.info("Trained: "+str(this))
            else:
                this.read()
                this.context.logger.info("Loaded: "+str(this))
        ##############################################################################

    def _to_retrain(self):
        retrain = self.local_config['parameters'].get('retrain', False)
        self.local_config['parameters']['retrain']= False
        return retrain

    def fit(self):
        self.real_fit()
        self.write()
        self.context.logger.info(str(self)+" saved.")

    def write(self):
        filepath = self.context.get_path(self)
       
        dump = {
            "model" : self.model,
            "config": self.local_config
        }
        
        with open(filepath, 'wb') as f:
          pickle.dump(dump, f)
      
    def read(self):
        dump_file = self.context.get_path(self)
        
        #TODO: manage the  if not file exist  case even if it is already managed by the general mecanism
        if self.saved:
            with open(dump_file, 'rb') as f:
                dump = pickle.load(f)
                self.model = dump['model']
                self.local_config = dump['config']
    
    @abstractmethod
    def init(self):
        pass
    
    @abstractmethod
    def check_configuration(self, local_config):
        pass

    @abstractmethod
    def real_fit(self):
        pass
    