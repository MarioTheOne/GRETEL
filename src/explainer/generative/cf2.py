import math
import os

import networkx as nx
import numpy as np
import torch
from dgl import from_networkx, to_networkx
from torch.utils.data import Dataset
from copy import deepcopy
# import wandb

from src.dataset.data_instance_base import DataInstance
from src.dataset.data_instance_features import DataInstanceWFeaturesAndWeights
from src.n_dataset.dataset_base import Dataset
from src.n_dataset.instances.graph import GraphInstance
from src.core.explainer_base import Explainer
from src.core.trainable_base import Trainable
from src.core.oracle_base import Oracle
from src.oracle.todo.oracle_cf2 import CustomDGLDataset


class CF2Explainer(Trainable, Explainer):

    def init(self):

        self.name = "cf2"
        self.n_nodes = 0
        self.converter = 0
        self.batch_size_ratio = 0
        self.fold_id = 0
        self.explainer_store_path = 0
        self.lr = 0
        self.weight_decay = 0
        self.gamma = 0
        self.lam = 0
        self.alpha = 0
        self.epochs = 0
        self.fold_id = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._fitted = False

        self.explainer = ExplainModelGraph(self.n_nodes).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.explainer.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def real_fit(self):
        self.explainer.train()

        for epoch in range(self.epochs):         
            losses = list()

            for graph in self.dataset.instances:
                pred1, pred2 = self.explainer(graph, self.oracle)
                loss = self.explainer.loss(graph,
                                           pred1, pred2,
                                           self.gamma, self.lam,
                                           self.alpha)
                
                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()
                self.optimizer.step()
            
            print(f"Epoch {epoch+1} --- loss {np.mean(losses)}")

    def explain(self, instance : GraphInstance):

        if(not self._fitted):
            self.explainer.train()
            self.fit()
            self._fitted = True

        self.explainer.eval()
        
        with torch.no_grad():
            cf_instance = deepcopy(instance)

            weighted_adj = self.explainer._rebuild_weighted_adj(g)
            masked_adj = self.explainer.get_masked_adj(weighted_adj).numpy()
            # update instance copy from masked_ajd
            cf_instance.data = masked_adj         
            print(f'Finished evaluating for instance {instance.id}')
            return cf_instance


class ExplainModelGraph(torch.nn.Module):
    
    def __init__(self, n_nodes: int):
        super(ExplainModelGraph, self).__init__()

        self.n_nodes = n_nodes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mask = self.build_adj_mask()

    def forward(self, graph, oracle):        
        pred1 = oracle.predict(graph)

        # re-build weighted adjacency matrix
        weighted_adj = self._rebuild_weighted_adj(graph)
        # get the masked_adj
        masked_adj = self.get_masked_adj(weighted_adj)
        # get the new weights as the difference between
        # the weighted adjacency matrix and the masked learned
        new_weights = weighted_adj - masked_adj
        # get only the edges that exist
        row_indices, col_indices = torch.where(new_weights != 0)

        cf_instance = deepcopy(graph)
        cf_instance.edge_weights = new_weights[row_indices, col_indices].detach().numpy()
        pred2 = oracle.predict(cf_instance)

        pred1 = torch.Tensor([pred1]).float()  # factual
        pred2 = torch.Tensor([pred2]).float()  # counterfactual

        return pred1, pred2

    def build_adj_mask(self):
        mask = torch.nn.Parameter(torch.FloatTensor(self.n_nodes, self.n_nodes))
        std = torch.nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (self.n_nodes + self.n_nodes)
        )
        with torch.no_grad():
            mask.normal_(1.0, std)
        return mask

    def get_masked_adj(self, weights):
        sym_mask = torch.sigmoid(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        masked_adj = weights * sym_mask
        return masked_adj

    def loss(self, graph, pred1, pred2, gam, lam, alp):
        weights = self._rebuild_weighted_adj(graph)
        bpr1 = torch.nn.functional.relu(gam + 0.5 - pred1)  # factual
        bpr2 = torch.nn.functional.relu(gam + pred2 - 0.5)  # counterfactual
        masked_adj = torch.flatten(self.get_masked_adj(weights))
        L1 = torch.linalg.norm(masked_adj, ord=1)
        return L1 + lam * (alp * bpr1 + (1 - alp) * bpr2)
    
    
    def _rebuild_weighted_adj(self, graph):
        u,v = graph.all_edges(order='eid')
        weights = np.zeros((self.n_nodes, self.n_nodes))
        weights[u.numpy(), v.numpy()] = graph.edata['weights'].detach().numpy()
        return torch.from_numpy(weights).float()