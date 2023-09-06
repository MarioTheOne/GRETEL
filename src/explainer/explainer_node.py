import sys
import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from src.oracle.todo.oracle_node_pt import NodeOracle
from src.dataset.data_instance_node import NodeDataInstance
from src.dataset.dataset_base import Dataset
from src.core.explainer_base import Explainer
from src.utils.cfgnnexplainer.utils import normalize_adj
from torch.nn.utils import clip_grad_norm
from torch_geometric.utils import dense_to_sparse
from src.utils.cfgnnexplainer.utils import get_degree_matrix, get_neighbourhood

from src.explainer.helpers.gcn_perturb import GCNSyntheticPerturb

class NodeExplainer(Explainer):

	def __init__(self, id, config_dict=None) -> None:
		super().__init__(id, config_dict)
		self._name = 'NodeExplainer'

	def explain(self, instance: NodeDataInstance, oracle: NodeOracle, dataset: Dataset):
		pass