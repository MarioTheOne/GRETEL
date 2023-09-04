from abc import ABCMeta, abstractmethod

from src.core.grtl_base import Base
from src.n_dataset.dataset_base import Dataset
from src.utils.context import Context


class Generator(Base, metaclass=ABCMeta):
    
    def __init__(self, context: Context, local_config, dataset=None) -> None:
        super().__init__(context, local_config)
        self.dataset = dataset
        self.init()
        self.current = 0
        
    @abstractmethod
    def get_num_instances(self):
        pass
    
    @abstractmethod
    def init():  
        pass
    
    @abstractmethod
    def generate_dataset(self):
        pass
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if (self.current + 1) < self.get_num_instances():
            self.current += 1
            return self.dataset.instances.get_instance(self.current)
        raise StopIteration
    
    def reset_iterator(self):
        self.current = 0