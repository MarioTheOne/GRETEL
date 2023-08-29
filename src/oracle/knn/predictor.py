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
        params = self.local_config['parameters']['model']['parameters']
        if "n_neighbors" in params:
            self.model = KNeighborsClassifier(**self.local_config['parameters']['model']['parameters'])
        else:
            self.model = KNeighborsClassifier(n_neighbors=3)

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

    '''def fit(self, dataset: Dataset, split_i=-1):
        # self.emb.fit(dataset)
        self._name = self._name + '_fit_on-' + dataset.name

        # If there is an available oracle trained on that dataset load it
        if os.path.exists(os.path.join(self._oracle_store_path, self.name, 'oracle.sav')):
            self.read_oracle(self._name)

        else: # If not then train
            self.inst_vectors = self.emb.get_embeddings()

            if split_i == -1:
                # Training with the entire dataset
                x = self.inst_vectors
                y = [ i.graph_label for i in dataset.instances]

                self.model.fit(x, y)
            else:
                # Training with an specific split
                spt = dataset.get_split_indices()[split_i]['train']

                x = [self.inst_vectors[i] for i in spt]
                y = [dataset.get_instance(i).graph_label for i in spt]

                self.model.fit(x, y)
            
            # Writing to disk the trained oracle
            self.write_oracle()'''

    def _real_predict(self, data_instance):
        return self.model.predict([data_instance])[0]

    def _real_predict_proba(self, data_instance):
        return self.model.predict_proba([data_instance])[0]

    def embedd(self, instance):
        return self.embedder.get_embedding(instance)

   ''' def write_oracle(self):
        oracle_path = os.path.join(self._oracle_store_path, self._name)
        os.mkdir(oracle_path)
        
        oracle_uri = os.path.join(oracle_path, 'oracle.sav')
        joblib.dump(self.clf, open(oracle_uri, 'wb'))

    def read_oracle(self, oracle_name):
        oracle_uri = os.path.join(self._oracle_store_path, oracle_name, 'oracle.sav')
        self.clf = joblib.load(open(oracle_uri, 'rb'))'''
