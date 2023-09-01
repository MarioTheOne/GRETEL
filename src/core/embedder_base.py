from abc import ABCMeta, abstractmethod
from src.core.trainable_base import Trainable

class Embedder(Trainable,metaclass=ABCMeta):

    '''def init_model(self):
        raise NotImplementedError()'''

    '''def real_fit(self):
        raise NotImplementedError()'''
    
    '''def check_configuration(self, local_config):
        raise NotImplementedError()'''
    
    @abstractmethod
    def get_embeddings(self):
        pass
    
    @abstractmethod
    def get_embedding(self, instance):
        pass
    


   