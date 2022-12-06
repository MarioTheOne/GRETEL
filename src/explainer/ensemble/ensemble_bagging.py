from typing import List
from src.explainer.ensemble.ensemble_base import EnsembleExplainer
from src.dataset.dataset_base import Dataset
from src.oracle.oracle_base import Oracle

from abc import ABC


class EnsembleBaggingExplainer(EnsembleExplainer):

    def __init__(self, id, config_dict=None, weak_explainers=None) -> None:
        super().__init__(id, config_dict, weak_explainers)
        self._name = 'ensemble_bagging_explainer'