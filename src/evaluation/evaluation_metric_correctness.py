from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.dataset.data_instance_base import DataInstance
from src.oracle.oracle_base import Oracle
from src.dataset.dataset_base import Dataset
from src.explainer.explainer_base import Explainer
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric




class CorrectnessMetric(EvaluationMetric):
    """Verifies that the class from the counterfactual example is different from that of the original instance
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Correctness'
        self._ged = GraphEditDistanceMetric()

    def evaluate(self, instance_1 : DataInstance, instance_2 : DataInstance, oracle : Oracle=None, explainer : Explainer=None, dataset : Dataset = None):

        label_instance_1 = oracle.predict(instance_1)
        label_instance_2 = oracle.predict(instance_2)
        oracle._call_counter -= 2

        ged = self._ged.evaluate(instance_1, instance_2, oracle)

        result = 1 if (label_instance_1 != label_instance_2) and (ged != 0) else 0
        
        return result