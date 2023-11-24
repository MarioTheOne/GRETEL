import copy
import sys
from abc import ABC

from src.core.explainer_base import Explainer
from src.explainer.ensemble.explanation_aggregator_base import ExplanationAggregator
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
import numpy as np


class ExplanationAggregatorTS(ExplanationAggregator):

    def __init__(self) -> None:
        self.ged = GraphEditDistanceMetric()
        pass

    def aggregate(self, org_instance, explanations, oracle):
        org_lbl = oracle.predict(org_instance)

        result = org_instance
        best_ged = sys.float_info.max
        for exp in explanations:
            exp_lbl = oracle.predict(exp)
            if exp_lbl != org_lbl:
                exp_dist = self.ged.evaluate(org_instance, exp)
                if exp_dist < best_ged:
                    best_ged = exp_dist
                    result = exp

        return result