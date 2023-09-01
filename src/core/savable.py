import os,pickle
from src.utils.context import Context
from abc import ABCMeta

class Savable(metaclass=ABCMeta):
    
    def __init__(self, context: Context, local_config) -> None:
        self.context:Context = context
        self.local_config = local_config

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

    def saved(self):
        return os.path.exists(self.context.get_path(self))
    
    @property
    def name(self):
        return self.context.get_name(self)
    
    def __str__(self):
        return self.name