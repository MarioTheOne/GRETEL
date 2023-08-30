from src.core.trainable_base import Trainable

class Embedder(Trainable):

    def init_model(self):
        raise NotImplementedError()

    def real_fit(self):
        raise NotImplementedError()
    
    def get_embeddings(self):
        raise NotImplementedError()

    def get_embedding(self, instance):
        raise NotImplementedError()
    
    def check_configuration(self, local_config):
        raise NotImplementedError()