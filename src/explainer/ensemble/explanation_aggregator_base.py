import copy
import sys
from abc import ABC

from src.core.explainer_base import Explainer
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
from src.evaluation.evaluation_metric_correctness import CorrectnessMetric
import numpy as np


class ExplanationAggregator(ABC):

    def __init__(self) -> None:
        pass

    def aggregate(self, org_instance, explanations, oracle):
        pass