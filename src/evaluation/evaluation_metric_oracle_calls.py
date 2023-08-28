from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.dataset.data_instance_base import DataInstance
from src.oracle.oracle_base import Oracle
from src.dataset.dataset_base import Dataset
from src.explainer.explainer_base import Explainer


class OracleCallsMetric(EvaluationMetric):
    """Provides the number of calls to the oracle an explainer has to perform in order to generate
    a counterfactual example
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Oracle_Calls'

    def evaluate(self, instance_1 : DataInstance, instance_2 : DataInstance, oracle : Oracle=None, explainer : Explainer=None, dataset : Dataset = None):
        result = oracle.get_calls_count()
        oracle.reset_call_count()
        return result