import copy
import sys

from src.core.explainer_base import Explainer
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
import numpy as np


class ExplainerEnsemble(Explainer):
    """The base class for the Explainer Ensemble. It should provide the common logic 
    for integrating multiple explainers and produce unified explanations"""

    def init(self):
        super().init()
        self.base_explainers = {}
        self.base_metrics = {}