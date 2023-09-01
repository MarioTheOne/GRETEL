from src.oracle.oracle_base import Oracle

import numpy as np

class TrianglesSquaresOracle(Oracle):

    def _real_predict(self, data_instance):
        # Classify
        if len(data_instance.graph.nodes) == 3 and len(data_instance.graph.edges) == 3:
            return 1 # triangle
        else:
            return 0 # other shape (squares)

    def _real_predict_proba(self, data_instance):
        # Classify
        if len(data_instance.graph.nodes) == 3 and len(data_instance.graph.edges) == 3:
            return np.array([[0,1]]) # triangle
        else:
            return np.array([[1,0]]) # other shape (squares)


    
