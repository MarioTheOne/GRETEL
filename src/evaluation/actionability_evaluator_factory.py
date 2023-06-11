from abc import ABC

from src.evaluation.actionability_evaluator_base import ActionabilityEvaluator
from src.evaluation.actionability_evaluator_bl import BLActionabilityEvaluator


class ActionabilityEvaluatorFactory:

    def __init__(self) -> None:
        self.actionability_evaluator_id_counter = 0

    
    def get_actionability_evaluator_by_name(self, ae_dict) -> ActionabilityEvaluator:
        ae_name = ae_dict['name']
        ae_parameters = ae_dict['parameters']

        if ae_name == 'dummy':
            return self.get_dummy_actionabilty_evaluator(ae_dict)
        
        elif ae_name == 'bl':
            restricted_nodes = ae_parameters['restricted_nodes']
            restricted_edges = ae_parameters['restricted_edges']
            restricted_node_features = ae_parameters['restricted_node_features']
            restricted_edge_features = ae_parameters['restricted_edge_features']
            return self.get_bl_actionability_evaluator(restricted_nodes, 
                                                       restricted_edges, 
                                                       restricted_node_features, 
                                                       restricted_edge_features,
                                                       ae_dict)
        
        else:
            raise ValueError('''The provided ActionabilityEvaluator name does not match any 
            Actionability Evaluator provided by the factory''')
        

    def get_dummy_actionabilty_evaluator(self, config_dict) -> ActionabilityEvaluator:
        ae = ActionabilityEvaluator(self.actionability_evaluator_id_counter, config_dict)
        self.actionability_evaluator_id_counter += 1
        return ae
    
    def get_bl_actionability_evaluator(self, 
                                       restricted_nodes, 
                                       restricted_edges, 
                                       restricted_node_features, 
                                       restricted_edge_features,
                                       config_dict):
        # Creating the Actionability Evaluator
        ae = BLActionabilityEvaluator(self.actionability_evaluator_id_counter, 
                                      restricted_nodes, 
                                      restricted_edges, 
                                      restricted_node_features, 
                                      restricted_edge_features, 
                                      config_dict)
        
        self.actionability_evaluator_id_counter += 1
        return ae
        
