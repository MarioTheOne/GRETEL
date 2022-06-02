from abc import ABC

from src.dataset.dataset_base import Dataset
from src.oracle.embedder_base import Embedder
from src.oracle.oracle_base import Oracle
from src.oracle.embedder_graph2vec import Graph2vec
from src.oracle.oracle_knn import KnnOracle
from src.oracle.oracle_svm import SvmOracle
from src.oracle.embedder_factory import EmbedderFactory
from src.oracle.oracle_asd_custom import ASDCustomOracle
from src.oracle.oracle_gcn_tf import TfGCNOracle

class OracleFactory(ABC):

    def __init__(self, oracle_store_path) -> None:
        super().__init__()
        self._oracle_store_path = oracle_store_path
        self._oracle_id_counter = 0

    def get_oracle_by_name(self, oracle_dict, dataset: Dataset, emb_factory: EmbedderFactory) -> Oracle:

        oracle_name = oracle_dict['name']
        oracle_parameters = oracle_dict['parameters']

        # Check if the oracle is a KNN classifier
        if oracle_name == 'knn':
            if not 'k' in oracle_parameters:
                raise ValueError('''The parameter "k" is required for knn''')
            if not 'embedder' in oracle_parameters:
                raise ValueError('''knn oracle requires an embedder''')

            emb = emb_factory.get_embedder_by_name(oracle_parameters['embedder'], dataset)

            return self.get_knn(dataset, emb, oracle_parameters['k'], -1, oracle_dict)

        # Check if the oracle is an SVM classifier
        elif oracle_name == 'svm':
            if not 'embedder' in oracle_parameters:
                raise ValueError('''svm oracle requires an embedder''')

            emb = emb_factory.get_embedder_by_name(oracle_parameters['embedder'], dataset)

            return self.get_svm(dataset, emb, -1, oracle_dict)

        # Check if the oracle is an ASD Custom Classifier
        elif oracle_name == 'asd_custom_oracle':
            return self.get_asd_custom_oracle(oracle_dict)

        # Check if the oracle is an ASD Custom Classifier
        elif oracle_name == 'gcn-tf':
            return self.get_gcn_tf(dataset, -1, oracle_dict)

        # If the oracle name does not match any oracle in the factory
        else:
            raise ValueError('''The provided oracle name does not match any oracle provided by the factory''')


    def get_knn(self, data: Dataset, embedder: Embedder, k, split_index=-1, config_dict=None) -> Oracle:
        embedder.fit(data)
        clf = KnnOracle(id=self._oracle_id_counter,oracle_store_path=self._oracle_store_path,  emb=embedder, k=k, config_dict=config_dict)
        self._oracle_id_counter +=1
        clf.fit(dataset=data, split_i=split_index)
        return clf

    def get_svm(self, data: Dataset, embedder: Embedder, split_index=-1, config_dict=None) -> Oracle:
        embedder.fit(data)
        clf = SvmOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, emb=embedder, config_dict=config_dict)
        self._oracle_id_counter +=1
        clf.fit(dataset=data, split_i=split_index)
        return clf

    def get_asd_custom_oracle(self, config_dict=None) -> Oracle:
        clf = ASDCustomOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, config_dict=config_dict)
        self._oracle_id_counter +=1
        return clf

    def get_gcn_tf(self, data: Dataset, split_index=-1, config_dict=None) -> Oracle:
        clf = TfGCNOracle(id=self._oracle_id_counter, oracle_store_path=self._oracle_store_path, config_dict=config_dict)
        self._oracle_id_counter +=1
        clf.fit(data, split_index)
        return clf