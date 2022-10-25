from src.dataset.dataset_base import Dataset
from src.oracle.oracle_base import Oracle

from abc import ABC


class Explainer(ABC):

    def __init__(self, id, config_dict=None) -> None:
        super().__init__()
        self._id = id
        self._name = 'abstract_explainer'
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

    def explain(self, instance, oracle: Oracle, dataset: Dataset):
        pass
