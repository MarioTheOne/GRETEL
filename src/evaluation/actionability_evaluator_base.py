import os
import pickle
import time
from abc import ABC

import jsonpickle
from scipy import rand

from src.dataset.data_instance_base import DataInstance
from src.dataset.dataset_base import Dataset
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle

class ActionabilityEvaluator(ABC):

    def __init__(self, id, config_dict):
        super().__init__()
        self.id = id
        self.name = 'abstract_actionability_evaluator'
        self.config_dict = config_dict


    def evaluate_graph(self, 
                       original_instance : DataInstance, 
                       cf_instance : DataInstance):
        return True


    def evaluate_edge(self,
                      original_instance : DataInstance, 
                      cf_instance : DataInstance, 
                      target_edge):
        return True


    def evaluate_node(self,
                      original_instance : DataInstance, 
                      cf_instance : DataInstance, 
                      target_node):
        return True


    def evaluate_node_features(self,
                               original_instance : DataInstance, 
                               original_features, 
                               new_features):
        return True
    

    def evaluate_edge_features(self,
                               original_instance : DataInstance, 
                               original_features, 
                               new_features):
        return True