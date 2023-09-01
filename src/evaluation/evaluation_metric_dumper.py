import os
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.dataset.data_instance_base import DataInstance
from src.core.oracle_base import Oracle
from src.dataset.dataset_base import Dataset
from src.explainer.explainer_base import Explainer
from src.evaluation.evaluation_metric_correctness import CorrectnessMetric

import jsonpickle


class InstancesDumper(EvaluationMetric):
    """As correctness measures if the algorithm is producing counterfactuals, but in Fidelity measures how faithful they are to the original problem,
     not just to the problem learned by the oracle. Requires a ground truth in the dataset
    """

    def __init__(self, conf_global, config_dict=None) -> None:
        super().__init__(config_dict)
        self._conf_global = conf_global
        self._name = 'Dumper'
        self._correctness = CorrectnessMetric()
        self._store_path = os.path.join(config_dict['parameters']['store_path'],self.__class__.__name__)
        if not os.path.exists(self._store_path):
            os.makedirs(self._store_path)
        

    def evaluate(self, instance_1 : DataInstance, instance_2 : DataInstance, oracle : Oracle=None, explainer : Explainer=None, dataset : Dataset = None):
        exp_path=os.path.join(self._store_path,dataset.name,explainer.name.replace('fold_id=.*_',''),str(explainer.fold_id))
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        correctness = self._correctness.evaluate(instance_1,instance_2,oracle)
        
        info = {
            "orginal_id":instance_1.id,
            "correctness":correctness,
             "fold": explainer.fold_id,
            "orginal":instance_1.to_numpy_array(),
            "counterfactual": instance_2.to_numpy_array()
        }
        with open(os.path.join(exp_path,str(instance_1.id)),'w') as dump_file:
            dump_file.write(jsonpickle.encode(info))
        
        return -1