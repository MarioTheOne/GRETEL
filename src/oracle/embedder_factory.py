from src.core.embedder_base import Embedder
from src.utils.utils import get_class

class EmbedderFactory:

    def __init__(self, embedder_store_path) -> None:
        self._embedder_id_count = 0
        self._embedder_store_path = embedder_store_path
        
    def get_embedder(self, context, embedder_snippet) -> Embedder:
        embedder = get_class(embedder_snippet['class'])(context, embedder_snippet)
        return embedder

    """def get_embedder_by_name(self, embedder_dict, dataset: Dataset) -> Embedder:
        
        embedder_name = embedder_dict['name']
        params_dict = embedder_dict['parameters']

        # Check the type of the mebedder
        if embedder_name == 'graph2vec':
            return self.get_graph2vec_embedder(dataset)
        if embedder_name == 'rdk_fingerprint':
            return self.get_rdk_fingerprint_embedder(dataset)
        else:
            raise ValueError('''The provided embedder name does not match any embedder 
                                provided by the factory''')

    def get_graph2vec_embedder(self, dataset: Dataset) -> Embedder:
        result = Graph2vec(self._embedder_id_count)
        self._embedder_id_count +=1
        result.fit(dataset)
        return result

    def get_rdk_fingerprint_embedder(self, dataset: Dataset) -> Embedder:
        result = RDKFingerprintEmbedder(self._embedder_id_count)
        self._embedder_id_count +=1
        return result"""