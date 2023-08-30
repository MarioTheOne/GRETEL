from src.core.embedder_base import Embedder
from src.utils.utils import get_class

class EmbedderFactory:
      
    def get_embedder(self, context, embedder_snippet) -> Embedder:
        embedder = get_class(embedder_snippet['class'])(context, embedder_snippet)
        self.context.logger.info("Created: "+ str(embedder))
        return embedder

    def __init__(self, context, local_conf=None) -> None:
        super().__init__()
        self.context = context
        self.local_conf = local_conf