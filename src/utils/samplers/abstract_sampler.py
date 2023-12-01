from abc import ABC, abstractmethod
from typing import List

from src.n_dataset.instances import DataInstance
from src.core.oracle_base import Oracle


class Sampler(ABC):
    
    def __init__(self):
        self._name = 'abstract_sampler'
    
        
    @abstractmethod
    def sample(self, instance: DataInstance, oracle: Oracle, **kwargs) -> List[DataInstance]:
        raise NotImplemented("The sample method hasn't been implemented yet!")