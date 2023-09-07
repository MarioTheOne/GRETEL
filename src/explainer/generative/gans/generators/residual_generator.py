from src.utils.torch.utils import rebuild_adj_matrix
import torch
import torch.nn as nn
from torch_geometric.nn import GAE

from src.utils.torch.gcn import GCN


class ResGenerator(nn.Module):
    
    def __init__(self, node_features, num_conv_layers=2, conv_booster=2, residuals=True):
        self.node_features = node_features
        self.num_conv_layers = num_conv_layers
        self.conv_booster = conv_booster
        self.residuals = residuals
        # encoder with no pooling
        self.encoder = GCN(self.node_features, self.num_conv_layers, self.conv_booster, nn.Identity())
        # graph autoencoder with inner product decoder
        self.model = GAE(encoder=self.encoder)
        
    def set_training(self, training):
        self.encoder.set_training(training)

    def forward(self, node_features, edge_list, edge_attr):
        encoded_node_features = self.model.encode(node_features, edge_list, edge_attr)
        edge_probabilities = self.model.decoder.forward_all(encoded_node_features, sigmoid=False)
        edge_probabilities = torch.nan_to_num(edge_probabilities, 0)

        if self.residuals:
            encoded_node_features = encoded_node_features.repeat(1, node_features.size(1) // encoded_node_features.size(1))
            encoded_node_features = torch.add(encoded_node_features, node_features)
            edge_probabilities += rebuild_adj_matrix(len(node_features), edge_list, edge_attr)
            
        return encoded_node_features, edge_list, torch.sigmoid(edge_probabilities)
    
    