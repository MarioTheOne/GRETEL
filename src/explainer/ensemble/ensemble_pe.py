from abc import ABC
from typing import List

from src.dataset.dataset_base import Dataset
from src.explainer.ensemble.ensemble_bagging import EnsembleBaggingExplainer
from src.oracle.oracle_base import Oracle


class PEEnsembleExplainer(EnsembleBaggingExplainer):

    def __init__(self, id, config_dict=None, weak_explainers=None) -> None:
        super().__init__(id, config_dict, weak_explainers)
        self._name = 'pe_ensemble'

    def explain(self, instance, oracle: Oracle, dataset: Dataset):
        pass
