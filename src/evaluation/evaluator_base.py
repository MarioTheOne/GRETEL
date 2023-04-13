import os
import pickle
import time
from abc import ABC

import jsonpickle
from scipy import rand
from typing_extensions import Self

from src.dataset.dataset_base import Dataset
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle
from src.utils.cfgnnexplainer.utils import safe_open


class Evaluator(ABC):

    def __init__(self, id, data: Dataset, oracle: Oracle, explainer: Explainer, evaluation_metrics, results_store_path, run_number=0) -> None:
        super().__init__()
        self._id = id
        self._name = 'Evaluator_for_' + explainer.name + '_using_' + oracle.name
        self._data = data
        self._oracle = oracle
        self._oracle.reset_call_count()
        self._explainer = explainer
        self._results_store_path = results_store_path
        self._evaluation_metrics = evaluation_metrics
        self._run_number = run_number
        self._explanations = []

        # Building the config file to write into disk
        evaluator_config = {'dataset': data._config_dict, 'oracle': oracle._config_dict, 'explainer': explainer._config_dict, 'metrics': []}
        for metric in evaluation_metrics:
            evaluator_config['metrics'].append(metric._config_dict)
        # creatig the results dictionary with the basic info
        self._results = {'config':evaluator_config, 'runtime': []}

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

    @property
    def dataset(self):
        return self._data

    @dataset.setter
    def dataset(self, new_dataset: Dataset):
        self._data = new_dataset

    @property
    def explanations(self):
        return self._explanations

    @explanations.setter
    def explanations(self, new_explanations_list):
        self._explanations = new_explanations_list


    def get_instance_explanation_pairs(self):
        # Check if the explanations were generated already
        if len(self.explanations) < 1:
            return None

        # iterates over the original instances and the explanations
        n_ins = len(self.dataset.instances)
        result = []
        for i in range(0, n_ins):
            result.append((self.dataset.instances[i], self.explanations[i]))

        return result


    def get_instance_and_counterfactual_classifications(self):
        # Check if the explanations were generated already
        if len(self.explanations) < 1:
            return None

        # iterates over the original instances and the explanations
        n_ins = len(self.dataset.instances)
        result = []
        for i in range(0, n_ins):
            label_inst = self._oracle.predict(self.dataset.instances[i])
            label_cf = self._oracle.predict(self.explanations[i])
            self._oracle._call_counter -= 2 

            result.append({'instance_id': self.dataset.instances[i].id,
                             'ground_truth_label': self.dataset.instances[i].graph_label,
                             'instance_label': label_inst,
                             'counterfactual_label': label_cf})

        return result


    def evaluate(self):
        for m in self._evaluation_metrics:
            self._results[m.name] = []

        # If the explainer was trained then evaluate only on the test set, else evaluate on the entire dataset
        fold_id = self._explainer.fold_id
        if fold_id == -1:
            for inst in self._data.instances:
                
                start_time = time.time()
                counterfactual = self._explainer.explain(inst, self._oracle, self._data)
                end_time = time.time()
                # giving the same id to the counterfactual and the original instance 
                counterfactual.id = inst.id
                self._explanations.append(counterfactual)

                # The runtime metric is built-in inside the evaluator``
                self._results['runtime'].append(end_time - start_time)

                self._real_evaluate(inst, counterfactual)
                print('evaluated instance with id ', str(inst.id))
        else:
            test_indices = self.dataset.splits[fold_id]['test']
            test_set = [i for i in self.dataset.instances if i.id in test_indices]

            for inst in test_set:
                
                start_time = time.time()
                counterfactual = self._explainer.explain(inst, self._oracle, self._data)
                end_time = time.time()
                # giving the same id to the counterfactual and the original instance 
                counterfactual.id = inst.id
                self._explanations.append(counterfactual)

                # The runtime metric is built-in inside the evaluator``
                self._results['runtime'].append(end_time - start_time)

                self._real_evaluate(inst, counterfactual)
                print('evaluated instance with id ', str(inst.id))

        self.write_results()


    def _real_evaluate(self, instance, counterfactual, oracle = None):
        is_alt = False
        if (oracle is None):
            is_alt = True
            oracle = self._oracle

        for metric in self._evaluation_metrics:
            m_result = metric.evaluate(instance, counterfactual, oracle)
            self._results[metric.name].append(m_result)


    def write_results(self):

        output_oracle_dataset_path = os.path.join(self._results_store_path, self._oracle.name)
        if not os.path.exists(output_oracle_dataset_path):
            os.mkdir(output_oracle_dataset_path)

        output_explainer_path = os.path.join(output_oracle_dataset_path, self._explainer.name)
        if not os.path.exists(output_explainer_path):
            os.mkdir(output_explainer_path)

        results_uri = os.path.join(output_explainer_path, 'results_run-' + str(self._run_number) + '.json')
        self._run_number += 1

        with open(results_uri, 'w') as results_writer:
            results_writer.write(jsonpickle.encode(self._results))

