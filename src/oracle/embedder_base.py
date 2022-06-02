from abc import ABC, abstractmethod
from src.dataset.dataset_base import Dataset


class Embedder(ABC):

    def __init__(self, id) -> None:
        super().__init__()
        self._id = id
        self._name = 'abstract_embedder'

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
    def fit(self, dataset : Dataset):
        pass

    @abstractmethod
    def get_embeddings(self):
        pass

    @abstractmethod
    def get_embedding(self, instance):
        pass