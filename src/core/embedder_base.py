from abc import ABCMeta, abstractmethod
from src.core.trainable_base import Trainable

class Embedder(Trainable,metaclass=ABCMeta):
        
    @abstractmethod
    def get_embeddings(self):
        pass
    
    @abstractmethod
    def get_embedding(self, instance):
        pass
    
    


   