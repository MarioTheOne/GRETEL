from src.core.factory_base import Factory
from src.core.embedder_base import Embedder


class EmbedderFactory(Factory):
    def get_embedder(self, embedder_snippet):        
        return self._get_object(embedder_snippet)
        

