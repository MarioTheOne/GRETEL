from src.dataset.dataset_base import Dataset
from src.oracle.embedder_base import Embedder
from src.oracle.oracle_base import Oracle

from sklearn.svm import LinearSVC
import os
import joblib

class SvmOracle(Oracle):

    def __init__(self, id, oracle_store_path, emb: Embedder, config_dict=None) -> None:
        super().__init__(id, oracle_store_path, config_dict)
        self.emb = emb
        self.clf =  LinearSVC()
        self.inst_vectors = None
        self._name = 'svm'
        

    def fit(self, dataset: Dataset, split_i=-1):
        # self.emb.fit(dataset)
        em = self.emb.get_embeddings()
        self._name = self._name + '_fit_on-' + dataset.name

        if os.path.exists(os.path.join(self._oracle_store_path, self.name, 'oracle.sav')):
            self.read_oracle(self._name)

        else:
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

            self.write_oracle()

    def _real_predict(self, data_instance):
        return self.clf.predict([data_instance])[0]

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
