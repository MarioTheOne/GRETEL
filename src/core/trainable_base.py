import os
from abc import ABC, abstractmethod
import pickle
import hashlib


class Trainable(ABC):
    
    def __init__(self, context, local_config) -> None:
        super().__init__()
        self.context = context
        self.local_config = local_config
        
        ##############################################################################
        # fit the model on a specific dataset
        # or read if already existing
        self.dataset = self.local_config['dataset']
        self.local_config['parameters']['fold_id'] =  self.local_config['parameters'].get('fold_id', -1)

        if('embedder' in self.local_config['parameters']):
            self.local_config['parameters']['embedder']['dataset'] = self.dataset
            self.local_config['parameters']['embedder']['fold_id'] = self.local_config['parameters']['fold_id']

        # real init details
        self.init()

        # retrain if explicitely specified or if the weights of the model don't exists
        if self.local_config['parameters'].get('retrain', False) or not os.path.exists(self.context.get_path(self)):
            self.context.logger.info(str(self)+" need to be trained.")
            self.fit()
        else:
            self.read()
            self.context.logger.info(str(self)+" loaded.")
        ##############################################################################
        
    def fit(self):
        self.real_fit()
        self.write()
        self.context.logger.info(str(self)+" saved.")
        
    @property
    def name(self):
        return self.context.get_name(self.__class__.__name__, self.local_config['parameters'])
    
    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def real_fit(self):
        pass
    
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
        
        if os.path.exists(dump_file):
            with open(dump_file, 'rb') as f:
                dump = pickle.load(f)
                self.model = dump['model']
                self.local_config = dump['config']
    
    @property
    def name(self):
        #return self.context.get_name(self.__class__.__name__, self.local_config['parameters'])
        data = self.context.get_name(self.__class__.__name__, self.local_config)
        md5_hash = hashlib.md5()
        md5_hash.update(data.encode('utf-8'))
        return  self.__class__.__name__+'-'+md5_hash.hexdigest()

    
    def __str__(self):
        return self.name
    