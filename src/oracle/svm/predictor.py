import os

import joblib
from sklearn.svm import LinearSVC

from src.oracle.oracle_base import Oracle


class SVMOracle(Oracle):

    def init(self):
        self.model = LinearSVC(**self.local_config['parameters']['model']['parameters'])
        
    def real_fit(self):
        fold_id = self.local_config['parameters'].get('fold_id', 0)
        embedding_snippet = self.local_config['parameters']['embedder']
        embedding_snippet['dataset'] = self.dataset
        
        self.embedder = self.context.factories['embedders'].get_embedder(self.context, embedding_snippet)
        inst_vectors = self.embedder.get_embeddings()

        if fold_id == -1:
            # Training with the entire dataset
            x = self.inst_vectors
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
        return self.model.predict_proba([data_instance])

    def embedd(self, instance):
        return self.embedder.get_embedding(instance)

    """def write(self):
        oracle_path = self.context.get_path(self)       
        joblib.dump(self.model, open(oracle_path, 'wb'))

    def read(self):
        self.model = joblib.load(open(self.context.get_path(self), 'rb'))"""
