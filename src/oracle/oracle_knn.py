from src.dataset.dataset_base import Dataset
from src.core.embedder_base import Embedder
from src.oracle.oracle_base import Oracle

from sklearn.neighbors import KNeighborsClassifier
import os
import joblib

class KnnOracle(Oracle):

    def __init__(self, id, oracle_store_path, emb: Embedder, k=3, config_dict=None) -> None:
        super().__init__(id, oracle_store_path, config_dict)
        self.emb = emb
        self.clf = KNeighborsClassifier(n_neighbors=k)
        self.inst_vectors = None
        self._name = 'knn_k-' + str(k)
        

    def fit(self, dataset: Dataset, split_i=-1):
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

                self.clf.fit(x, y)
            else:
                # Training with an specific split
                spt = dataset.get_split_indices()[split_i]['train']

                x = [self.inst_vectors[i] for i in spt]
                y = [dataset.get_instance(i).graph_label for i in spt]

                self.clf.fit(x, y)
            
            # Writing to disk the trained oracle
            self.write_oracle()

    def _real_predict(self, data_instance):
        return self.clf.predict([data_instance])[0]

    def _real_predict_proba(self, data_instance):
        return self.clf.predict_proba([data_instance])[0]

    def embedd(self, instance):
        return self.emb.get_embedding(instance)

    def write_oracle(self):
        oracle_path = os.path.join(self._oracle_store_path, self._name)
        os.mkdir(oracle_path)
        
        oracle_uri = os.path.join(oracle_path, 'oracle.sav')
        joblib.dump(self.clf, open(oracle_uri, 'wb'))

    def read_oracle(self, oracle_name):
        oracle_uri = os.path.join(self._oracle_store_path, oracle_name, 'oracle.sav')
        self.clf = joblib.load(open(oracle_uri, 'rb'))

