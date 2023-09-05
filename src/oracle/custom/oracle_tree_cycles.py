from src.oracle.custom.custom_base import CustomOracle

import numpy as np
import networkx as nx

class TreeCyclesOracle(CustomOracle):

    def _real_predict(self, data_instance):
        try:
            nx.find_cycle(data_instance.graph, orientation='ignore')
            return 1
        except nx.exception.NetworkXNoCycle:
            return 0
        
    def _real_predict_proba(self, data_instance):
        # softmax-style probability predictions
        try:
            nx.find_cycle(data_instance.graph, orientation='ignore')
            return np.array([0,1])
        except nx.exception.NetworkXNoCycle:
            return np.array([1,0])
        
    