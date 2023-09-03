import os
from src.core.grtl_base import Base

from abc import ABCMeta,abstractmethod

class Savable(Base,metaclass=ABCMeta):
    
    @abstractmethod 
    def write(self):
        pass

    @abstractmethod 
    def read(self):
        pass

    def saved(self):
        return os.path.exists(self.context.get_path(self))
    
   