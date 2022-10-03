from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.explainer.explainer_base import Explainer
from src.dataset.dataset_base import Dataset
from src.oracle.oracle_base import Oracle
from src.dataset.data_instance_base import DataInstance
import tensorflow as tf

class CffExplainer(Explainer):
    """
    Conterfactual factual explainer
    """

    def __init__(self,id, instance_distance_function : EvaluationMetric, config_dict=None) -> None:
        super().__init__(id, config_dict)
        self._gd = instance_distance_function
        self._name = 'CffExplainer'

    def explain(self, instance:DataInstance, oracle: Oracle, dataset: Dataset):
        opt_problem = RelaxedProblem(oracle, instance)

class RelaxedProblem(tf.Module):
    def __init__(self,instance:DataInstance,oracle:Oracle):
        self.instance = instance
        self.oracle = oracle
        self.current_pred = oracle.predict(instance)
        self.num_nodes = len(instance.node_labels)
        self.adj_mask = self.construct_adj_mask()
        self.diag_mask = tf.ones(self.num_nodes, self.num_nodes) - tf.eye(self.num_nodes)
    
    def construct_adj_mask(self):
        #mask = torch.nn.Parameter(torch.FloatTensor(self.num_nodes, self.num_nodes))
        #std = torch.nn.init.calculate_gain("relu") * math.sqrt(
        #    2.0 / (self.num_nodes + self.num_nodes)
        #)
        #with torch.no_grad():
        #    mask.normal_(1.0, std)
        std = (2/(self.num_nodes))**0.5 # gain of the relu function is all  2**0.5 (LOL)
        mask = tf.Variable(tf.random.normal([self.num_nodes, self.num_nodes], mean=1.0, stddev=std))
        return mask

    def get_masked_adj(self):
        sym_mask = tf.sigmoid(self.adj_mask)
        sym_mask = (sym_mask + tf.transpose(sym_mask)) / 2
        adj = self.instance.graph.adj 
        flatten_sym_mask = tf.reshape(sym_mask, (-1, ))
        masked_adj = adj * flatten_sym_mask
        return masked_adj