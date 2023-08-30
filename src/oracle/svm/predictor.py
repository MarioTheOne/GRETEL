from sklearn.svm import LinearSVC

from src.oracle.oracle_base import Oracle


class SVMOracle(Oracle):

    def init(self):
        self.model = LinearSVC(**self.local_config['parameters']['model']['parameters'])
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
        return self.model.predict(data_instance)
    
    def _real_predict_proba(self, data_instance):
        return self.model._predict_proba_lr(data_instance).squeeze()

    def embedd(self, instance):
        return self.embedder.get_embedding(instance).reshape(1,-1)