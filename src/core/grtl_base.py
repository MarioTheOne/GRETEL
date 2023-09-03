from abc import ABCMeta
from src.utils.context import Context
import copy



class Base(metaclass=ABCMeta):
    
    def __init__(self, context: Context, local_config=None) -> None:
        super().__init__()
        self.context:Context = context
        self.local_config = copy.deepcopy(local_config)

    @property
    def name(self):
        return self.context.get_name(self)
    
    def __str__(self):
        return self.name