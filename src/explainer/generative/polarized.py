import copy
import numpy as np
import random

from src.core.explainer_base import Explainer
from src.core.trainable_base import Trainable
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
from src.n_dataset.instances.graph import GraphInstance

class PolarizedExplainer(Explainer, Trainable):

    def init(self):
        self._fitted = False
        self.polarized = {}
        pass

    def explain(self, instance : GraphInstance):
        
        if(not self._fitted):
            self.real_fit()

        l_class = self.oracle.predict(instance)
        matrix = copy.deepcopy(instance.data)
        polarized = copy.deepcopy(self.polarized[instance.id])
        
        while len(np.nonzero(polarized)):
            next_edge = np.unravel_index(np.argmax(polarized, axis=None), polarized.shape)
            value = polarized[next_edge]

            selector = random.uniform(0,1)
            # take or not the edge
            if selector < abs(value):
                # add edge
                if value > 0:
                    matrix[next_edge] = 1
                # remove the edge
                elif value < 0:
                    matrix[next_edge] = 0

            # test for counterfactual
            generated = GraphInstance(instance.id, instance.label, matrix)
            r_predict = self.oracle.predict(generated)

            # todo check for new solution instead of a previos seen one

            # new non seen instance
            if r_predict != l_class:
                return generated
        
        return instance


    
    def real_fit(self):
        for inst in self.dataset.instances:
            # oracle result
            l_input_inst = self.oracle.predict(inst)
            # original edges
            edges = np.nonzero(inst.data)
            # initialize weights 
            adj = copy.deepcopy(inst.data)
            adj[edges] = 1

            for target in self.dataset.instances:
                if target.id == inst.id:
                    continue
                
                # skip non counterfactual examples
                r_input_inst = self.oracle.predict(target)
                if l_input_inst == r_input_inst:
                    continue

                t_data = copy.deepcopy(target.data)
                t_data[np.nonzero(t_data)] = 1

                for x in range(adj.shape(0)):
                    for y in range(adj.shape(1)):

                        if x > t_data.shape(0) or y > t_data.shape(1):
                            pass

                        # absent edge on the cf
                        if inst.data[x,y] > 0 and target.data[x,y] == 0:
                            adj[x,y] -= 1
                        # absent edge on the original
                        elif inst.data[x,y] == 0 and target.data[x,y] > 0:
                            adj[x,y] += 1
                
                #todo normalize matrix
                self.polarized[id] = adj
                
                



