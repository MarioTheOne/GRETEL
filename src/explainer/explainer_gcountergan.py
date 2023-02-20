from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.explainer.explainer_base import Explainer
from src.dataset.dataset_base import Dataset
from src.oracle.oracle_base import Oracle
from src.dataset.data_instance_base import DataInstance

import numpy as np
import json
import pickle
import time
import math
from sklearn.model_selection import train_test_split
import os

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torch
import os
import numpy as np
import json
import math
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GAE


class Discriminator(nn.Module):

  def __init__(self, n_nodes=28, n_features=1, batch_size=32, device='cuda'):
    """This class provides a GCN to discriminate between real and generated graph instances"""
    super(Discriminator, self).__init__()

    self.training = False
    self.device = device
    self.batch_size = batch_size
    self.n_nodes = n_nodes

    self.conv1 = GCNConv(n_features, 64)
    self.conv2 = GCNConv(64, 64)
    self.conv3 = GCNConv(64, 64)
    self.flatten = nn.Flatten()
    self.fc = nn.Linear(self.n_nodes * 64, 1)


  def set_training(self, training):
    self.training = training

  def forward(self, x, edge_list):
    x = self.conv1(x, edge_list)

    if self.training:
      x = self.add_gaussian_noise(x)

    x = F.leaky_relu(x, negative_slope=.2)
    x = F.dropout(x, p=.4, training=self.training)
    x = self.conv2(x, edge_list)
    x = F.leaky_relu(x, negative_slope=.2)
    x = F.dropout(x, p=.4, training=self.training)
    x = self.conv3(x, edge_list)
    x = F.leaky_relu(x, negative_slope=.2)
    x = F.dropout(x, p=.4, training=self.training).view(self.batch_size, self.n_nodes, -1)
    x = self.flatten(x)
    x = self.fc(x)
    x = torch.sigmoid(x).squeeze()

    return x

  def add_gaussian_noise(self, x, sttdev=0.2):
    noise = torch.randn(x.size(), device=self.device).mul_(sttdev)
    return x + noise
  

class GCNGeneratorEncoder(nn.Module):

  def __init__(self, in_channels=1, out_channels=64):
    super().__init__()
    self.conv1 = GCNConv(in_channels, out_channels)
    self.conv2 = GCNConv(out_channels, out_channels)
    self.conv3 = GCNConv(out_channels, in_channels)
    
    self.training = False

  def forward(self, x, edge_index):
    x = self.conv1(x, edge_index)
    x = F.leaky_relu(x, negative_slope=.2)
    x = F.dropout(x, p=.2, training=self.training)
    x = self.conv2(x, edge_index)
    x = F.leaky_relu(x, negative_slope=.2)
    x = F.dropout(x, p=.2, training=self.training)
    x = self.conv3(x, edge_index)
    x = torch.tanh(x)
    return x

  def set_training(self, training):
    self.training = training
  

class ResidualGenerator(nn.Module):

  def __init__(self, residuals=True, threshold=0.5):
    super(ResidualGenerator, self).__init__()

    self.gcn_encoder = GCNGeneratorEncoder(1,64)
    self.model = GAE(encoder=self.gcn_encoder)
    self.residuals = residuals
    self.threshold = torch.Tensor([threshold])

  def set_training(self, training):
    self.gcn_encoder.set_training(training)
    
  def forward(self, node_features, edge_list):
    encoded_node_features = self.model.encode(node_features, edge_list)
    edge_probabilities = self.model.decode(encoded_node_features, edge_list)

    if self.residuals:
      encoded_node_features = torch.add(encoded_node_features, node_features)
    
    edge_probabilities = (edge_probabilities > self.threshold).float()

    return encoded_node_features, edge_probabilities

