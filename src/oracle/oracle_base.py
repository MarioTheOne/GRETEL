from src.dataset.dataset_base import Dataset

from abc import ABC, abstractmethod


class Oracle(ABC):

    def __init__(self, id, oracle_store_path, config_dict=None) -> None:
        super().__init__()
        self._call_counter = 0
        self._id = id
        self._name = 'abstract_oracle'
        self._oracle_store_path = oracle_store_path
        self._config_dict = config_dict

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, new_id):
        self._id = new_id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @abstractmethod
    def fit(self, dataset: Dataset, split_i=0):
        pass

    
    def predict(self, data_instance):
        """predicts the label of a given data instance
        -------------
        INPUT:
            data_instance : The instance whose class is going to be predicted 
        -------------
        OUTPUT:
            The predicted label for the data instance
        """
        self._call_counter += 1

        return self._real_predict(self.embedd(data_instance))

    
    def predict_proba(self, data_instance):
        """predicts the probability estimates for a given data instance
        -------------
        INPUT:
            data_instance : The instance whose class is going to be predicted 
        -------------
        OUTPUT:
            The predicted probability estimates for the data instance
        """
        self._call_counter += 1

        return self._real_predict_proba(self.embedd(data_instance))
    
    def predict_list(self, dataset: Dataset, split_i=0):

        sptest = dataset.get_split_indices()[split_i]['test']

        result = [self.predict(dataset.get_instance(i)) for i in sptest]

        return result

    @abstractmethod
    def _real_predict(self, data_instance):
        pass
    
    @abstractmethod
    def _real_predict_proba(self, data_instance):
        pass
    
    @abstractmethod
    def embedd(self, instance):
        pass

    def get_calls_count(self):
        return self._call_counter

    def reset_call_count(self):
        self._call_counter = 0

    @abstractmethod
    def read_oracle(self, oracle_name):
        pass

    @abstractmethod
    def write_oracle(self):
        pass

