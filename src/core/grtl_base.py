from abc import ABCMeta, abstractmethod
from src.utils.context import Context
import copy

class Base(metaclass=ABCMeta):
    
    def __init__(self, context: Context, local_config=None) -> None:
        super().__init__()
        self.context:Context = context
        self.local_config = copy.deepcopy(local_config) if local_config else None
        #TODO: Move here the call of check_configuration
        
    @property
    def name(self):
        return self.context.get_name(self)
    
    def __str__(self):
        return self.name
    
    def check_configuration(self):
        self.local_config['parameters'] = self.local_config.get('parameters',{})

