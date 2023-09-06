from src.dataset.data_instance_base import DataInstance
from src.dataset.dataset_base import Dataset
from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle

import networkx as nx

from abc import ABC, abstractmethod

class EvaluationMetric(ABC):

    def __init__(self, config_dict=None) -> None:
        super().__init__()
        self._name = 'abstract_metric'
        self._config_dict = config_dict

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @abstractmethod
    def evaluate(self, instance_1 : DataInstance, instance_2 : DataInstance, oracle : Oracle=None, explainer : Explainer=None, dataset : Dataset = None):
        pass
    