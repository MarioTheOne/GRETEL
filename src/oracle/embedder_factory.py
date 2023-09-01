from src.core.embedder_base import Embedder
from src.utils.utils import get_class
from src.utils.context import Context


class EmbedderFactory:

    def __init__(self, context:Context, local_conf=None) -> None:
        super().__init__()
        self.contex:Context = context
        self.local_conf = local_conf
      
    def get_embedder(self, context, embedder_snippet) -> Embedder:
        embedder = get_class(embedder_snippet['class'])(context, embedder_snippet)
        self.context.logger.info("Created: "+ str(embedder))
        return embedder

