from abc import ABC, abstractmethod
from typing import List

from src.dataset.data_instance_base import DataInstance
from src.oracle.oracle_base import Oracle


class Sampler(ABC):
    
    def __init__(self):
        self._name = 'abstract_sampler'
    
        
    @abstractmethod
    def sample(self, instance: DataInstance, oracle: Oracle, **kwargs) -> List[DataInstance]:
        raise NotImplemented("The sample method hasn't been implemented yet!")