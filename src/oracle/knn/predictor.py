from src.dataset.dataset_base import Dataset
from src.core.embedder_base import Embedder
from src.oracle.oracle_base import Oracle

from sklearn.neighbors import KNeighborsClassifier
import os
import joblib

class KNNOracle(Oracle):

    '''def __init__(self, id, oracle_store_path, emb: Embedder, k=3, config_dict=None) -> None:
        super().__init__(id, oracle_store_path, config_dict)
        self.emb = emb
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.inst_vectors = None'''

    def init(self):
        self.model = KNeighborsClassifier(**self.local_config['parameters']['model']['parameters'])         
        embedding_snippet = self.local_config['parameters']['embedder']             
        self.embedder = self.context.factories['embedders'].get_embedder(self.context, embedding_snippet)

    def real_fit(self):
        fold_id = self.local_config['parameters'].get('fold_id', -1)
       
        inst_vectors = self.embedder.get_embeddings()

        if fold_id == -1:
            # Training with the entire dataset
            x = inst_vectors
            y = [ i.graph_label for i in self.dataset.instances]
        else:
            # Training with an specific split
            spt = self.dataset.get_split_indices()[fold_id]['train']
            x = [inst_vectors[i] for i in spt]
            y = [self.dataset.get_instance(i).graph_label for i in spt]

        self.model.fit(x, y)

    def _real_predict(self, data_instance):
        return self.model.predict([data_instance])[0]

    def _real_predict_proba(self, data_instance):
        return self.model.predict_proba([data_instance])[0]

    def embedd(self, instance):
        return self.embedder.get_embedding(instance)

    def check_configuration(self, local_config):
        params = local_config['parameters']['model']['parameters']
        if "n_neighbors" not in params:
            params['n_neighbors'] = 3
        return params