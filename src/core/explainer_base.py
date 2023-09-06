from src.core.oracle_base import Oracle
from src.core.grtl_base import Base
from src.n_dataset.dataset_base import Dataset

from abc import ABCMeta, abstractmethod


class Explainer(Base, metaclass=ABCMeta):

    
    @abstractmethod
    def explain(self, instance, oracle: Oracle, dataset: Dataset):
        pass
