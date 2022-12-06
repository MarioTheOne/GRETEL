import sys
import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from src.explainer.explainer_node import NodeExplainer
from src.oracle.oracle_node_pt import NodeOracle
from src.dataset.data_instance_node import NodeDataInstance
from src.dataset.dataset_base import Dataset
from src.explainer.explainer_base import Explainer
from src.utils.cfgnnexplainer.utils import normalize_adj
from torch.nn.utils import clip_grad_norm
from torch_geometric.utils import dense_to_sparse
from src.utils.cfgnnexplainer.utils import get_degree_matrix, get_neighbourhood

from src.explainer.helpers.gcn_perturb import GCNSyntheticPerturb

EPS = 1e-15

class CFExplainer:
	"""
	CF Explainer class, returns counterfactual subgraph
	"""
	def __init__(self, sub_adj, sub_feat, n_hid, dropout,
				  sub_labels, y_pred_orig, num_classes, beta, device, oracle: NodeOracle = None):
		super(CFExplainer, self).__init__()
		self._name = 'cfgnnexplainer'
		self.oracle = oracle
		self.oracle.eval()
		
		self.sub_adj = sub_adj
		self.sub_feat = sub_feat
		self.n_hid = n_hid
		self.dropout = dropout
		self.sub_labels = sub_labels
		self.y_pred_orig = y_pred_orig
		self.beta = beta
		self.num_classes = num_classes
		self.device = device

		self.cuda_enabled = self.device == "cuda"
		
		# Instantiate CF model class, load weights from original model
		self.cf_model = GCNSyntheticPerturb(self.sub_feat.shape[1], n_hid, n_hid,
											self.num_classes, self.sub_adj, dropout, beta, cuda_enabled=self.cuda_enabled)

		self.cf_model.load_state_dict(self.oracle.state_dict(), strict=False)

		# Freeze weights from original model in cf_model
		for name, param in self.cf_model.named_parameters():
			if name.endswith("weight") or name.endswith("bias"):
				param.requires_grad = False
		for name, param in self.oracle.named_parameters():
			print("orig model requires_grad: ", name, param.requires_grad)
		for name, param in self.cf_model.named_parameters():
			print("cf model requires_grad: ", name, param.requires_grad)


	def explain(self, cf_optimizer, node_idx, new_idx, lr, n_momentum, num_epochs):
		self.node_idx = node_idx
		self.new_idx = new_idx

		self.x = self.sub_feat
		self.A_x = self.sub_adj

		if (self.cuda_enabled):
			self.x = self.x.cuda()
			self.A_x = self.A_x.cuda()

		self.D_x = get_degree_matrix(self.A_x, self.cuda_enabled)

		if cf_optimizer == "SGD" and n_momentum == 0.0:
			self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
		elif cf_optimizer == "SGD" and n_momentum != 0.0:
			self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum)
		elif cf_optimizer == "Adadelta":
			self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)

		best_cf_example = []
		best_loss = np.inf
		num_cf_examples = 0
		for epoch in range(num_epochs):
			new_example, loss_total = self.train(epoch)
			if new_example != [] and loss_total < best_loss:
				best_cf_example.append(new_example)
				best_loss = loss_total
				num_cf_examples += 1
		print("{} CF examples for node_idx = {}".format(num_cf_examples, self.node_idx))
		print(" ")
		return(best_cf_example)


	def train(self, epoch):
		self.cf_model.train()
		self.cf_optimizer.zero_grad()

		# output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
		# output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
		output = self.cf_model.forward(self.x, self.A_x)
		output_actual, self.P = self.cf_model.forward_prediction(self.x)

		# Need to use new_idx from now on since sub_adj is reindexed
		y_pred_new = torch.argmax(output[self.new_idx])
		y_pred_new_actual = torch.argmax(output_actual[self.new_idx])
		
		
		if (self.cuda_enabled):
			self.x = self.x.cuda()
			self.A_x = self.A_x.cuda()

		# loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
		loss_total, loss_pred, loss_graph_dist, cf_adj = self.cf_model.loss(output[self.new_idx], self.y_pred_orig, y_pred_new_actual)
		loss_total.backward()
		clip_grad_norm(self.cf_model.parameters(), 2.0)

		a = time.time()
		self.cf_optimizer.step()
		b = time.time()
		print(f"optimizer step time: {(b - a) / 1000} seconds")

		print('Node idx: {}'.format(self.node_idx),
			  'New idx: {}'.format(self.new_idx),
			  'Epoch: {:04d}'.format(epoch + 1),
			  'loss: {:.4f}'.format(loss_total.item()),
			  'pred loss: {:.4f}'.format(loss_pred.item()),
			  'graph loss: {:.4f}'.format(loss_graph_dist.item()))
		print('Output: {}\n'.format(output[self.new_idx].data),
			  'Output nondiff: {}\n'.format(output_actual[self.new_idx].data),
			  'orig pred: {}, new pred: {}, new pred nondiff: {}'.format(self.y_pred_orig, y_pred_new, y_pred_new_actual))
		print(" ")
		cf_stats = []
		if y_pred_new_actual != self.y_pred_orig:
			# Modified version of cf_stats. New version returns the resulting feature matrix as an extra value in the stats.
			cf_stats = [self.node_idx, self.new_idx,
						cf_adj.detach().numpy(), self.sub_adj.detach().numpy(),
						self.y_pred_orig.item(), y_pred_new.item(),
						y_pred_new_actual.item(), self.sub_labels[self.new_idx].numpy(),
						self.sub_adj.shape[0], loss_total.item(),
						loss_pred.item(), loss_graph_dist.item(), self.x.detach().numpy()]



		return(cf_stats, loss_total.item())


class CFGNNExplainer(NodeExplainer):

	def __init__(self, id, config_dict=None) -> None:
		super().__init__(id, config_dict)
		self._name = 'CFGNNExplainer'

		enable_params = config_dict is not None and "parameters" in config_dict.keys()
		param_dict = config_dict["parameters"] if enable_params else None

		self._hidden = param_dict["hidden"] if enable_params and "hidden" in param_dict.keys() else 20 
		self._n_layers = param_dict["n_layers"] if enable_params and "n_layers" in param_dict.keys() else 3
		self._dropout = param_dict["dropout"] if enable_params and "hiddropoutden" in param_dict.keys() else 0.0
		self._seed = param_dict["seed"] if enable_params and "seed" in param_dict.keys() else 42
		self._lr = param_dict["lr"] if enable_params and "lr" in param_dict.keys() else 0.005
		self._optimizer = param_dict["optimizer"] if enable_params and "optimizer" in param_dict.keys() else "SGD"
		self._n_momentum = param_dict["n_momentum"] if enable_params and "n_momentum" in param_dict.keys() else 0.0
		self._beta = param_dict["beta"] if enable_params and "beta" in param_dict.keys() else 0.5
		self._num_epochs = param_dict["num_epochs"] if enable_params and "num_epochs" in param_dict.keys() else 500
		self._device = param_dict["device"] if enable_params and "device" in param_dict.keys() else 'cpu'
		self._clip = param_dict["clip"] if enable_params and "clip" in param_dict.keys() else 2.0
		self._weight_decay = param_dict["weight_decay"] if enable_params and "weight_decay" in param_dict.keys() else 0.001


	def explain(self, instance: NodeDataInstance, oracle: NodeOracle, dataset: Dataset):
		np.random.seed(self._seed)
		torch.manual_seed(self._seed)
		torch.autograd.set_detect_anomaly(True)

		data = instance.graph_data
		adj = torch.Tensor(data["adj"]).squeeze()
		features = torch.Tensor(data["feat"]).squeeze()
		labels = torch.tensor(data["labels"]).squeeze()
		idx_train = torch.tensor(data["train_idx"])
		idx_test = torch.tensor(data["test_idx"])
		edge_index = dense_to_sparse(adj)       # Needed for pytorch-geo functions
		norm_adj = normalize_adj(adj)       # According to reparam trick from GCN paper

		# if the method does not find a counterfactual example returns the original graph
		min_counterfactual = instance

		y_pred_orig = oracle.predict_all_nodes(instance)
		print("y_true counts: {}".format(np.unique(labels.numpy(), return_counts=True)))
		print("y_pred_orig counts: {}".format(np.unique(y_pred_orig.numpy(), return_counts=True)))      # Confirm model is actually doing something

		model = oracle.get_torch_model()

		i = instance.target_node
		sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(int(i), edge_index, self._n_layers + 1, features, labels)
		new_idx = node_dict[int(i)]

		explainer = CFExplainer(oracle=oracle,
								sub_adj=sub_adj,
								sub_feat=sub_feat,
								n_hid=self._hidden,
								dropout=self._dropout,
								sub_labels=sub_labels,
								y_pred_orig=y_pred_orig[i],
								num_classes=len(labels.unique()),
								beta=self._beta,
								device=self._device)

		if self._device == 'cuda':
			model.cuda()
			explainer.cf_model.cuda()
			adj = adj.cuda()
			norm_adj = norm_adj.cuda()
			features = features.cuda()
			labels = labels.cuda()
			idx_train = idx_train.cuda()
			idx_test = idx_test.cuda()
			
		min_counterfactual = explainer.explain(node_idx=i, cf_optimizer=self._optimizer, new_idx=new_idx, lr=self._lr,
								   n_momentum=self._n_momentum, num_epochs=self._num_epochs)

		#build the counterfactual datainstance
		counterfactual_instance = NodeDataInstance()
		counterfactual_instance.target_node = instance.target_node
		new_data = data.copy()
		if (len(min_counterfactual) > 0):
			new_data["adj"] = min_counterfactual[0][3]
			new_data["feat"] = min_counterfactual[0][12]
			new_data["labels"] = sub_labels
			counterfactual_instance.target_node = new_idx

		counterfactual_instance.graph_data = new_data
		counterfactual_instance.graph = self.generate_networkx(min_counterfactual, adj)

		return counterfactual_instance

	def generate_networkx(self, counterfactual, original_adj):
		return get_nx_from_edges_list(counterfactual[0][3]) if len(counterfactual) > 0 else get_nx_from_edges_list(original_adj)


def get_nx_from_edges_list(adj_matrix):
	edges_list = []
	for node_idx in range(len(adj_matrix)):
		node_adj = adj_matrix[node_idx]
		for other_node_idx in range(len(node_adj)):
			if (node_adj[other_node_idx] > 0):
				edges_list.append((node_idx, other_node_idx))

	return nx.Graph(edges_list)
