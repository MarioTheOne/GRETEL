from src.dataset.data_instance_target_node import TargetNodeDataInstance
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.dataset.data_instance_base import DataInstance
from src.oracle.oracle_base import Oracle


class FidelityNodeMetric(EvaluationMetric):
    """As correctness measures if the algorithm is producing counterfactuals, but in Fidelity measures how faithful they are to the original problem,
     not just to the problem learned by the oracle. Requires a ground truth in the dataset
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'FidelityNode'

    def evaluate(self, instance_1: TargetNodeDataInstance, instance_2: DataInstance, oracle: Oracle):

        label_instance_1 = oracle.predict(instance_1)
        label_instance_2 = oracle.predict(instance_2)
        oracle._call_counter -= 2

        prediction_fidelity = 1 if (label_instance_1 == instance_1.node_labels.get(instance_1.target_node)) else 0
        
        counterfactual_fidelity = 1 if (label_instance_2 == instance_1.node_labels.get(instance_1.target_node)) else 0

        result = prediction_fidelity - counterfactual_fidelity
        
        return result