from src.dataset.data_instance_base import DataInstance

from abc import ABC


class CounterfactualExplanation(ABC):

    def __init__(self, ranked_cf_instances=None) -> None:
        super().__init__()
        self._ranked_cf_instances = ranked_cf_instances

    def top_counterfactual(self) -> DataInstance:
        return self._ranked_cf_instances[0]
    
    def ranked_counterfactuals(self):
        return self._ranked_cf_instances
