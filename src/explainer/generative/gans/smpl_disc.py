import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.utils.cfg_utils import default_cfg


class SimpleDiscriminator(nn.Module):
    
    def __init__(self, num_nodes, node_features):
        """This class provides a GCN to discriminate between real and generated graph instances"""
        super(SimpleDiscriminator, self).__init__()

        self.training = False
        
        self.conv1 = GCNConv(node_features, 2).double()
        self.fc = nn.Linear(num_nodes * 2, 1).double()
        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        
    def set_training(self, training):
        self.training = training

    def forward(self, x, edge_list, edge_attr):
        x = x.double()
        edge_attr = edge_attr.double()
        x = self.conv1(x, edge_list, edge_attr)
        
        if self.training:
            x = self.add_gaussian_noise(x)

        x = F.relu(x)
        x = F.dropout(x, p=.4, training=self.training)
        x = torch.flatten(x)
        x = self.fc(x)
        x = torch.sigmoid(x).squeeze()

        return x
    
    def add_gaussian_noise(self, x, sttdev=0.2):
        noise = torch.randn(x.size(), device=self.device).mul_(sttdev)
        return x + noise
        
    @default_cfg
    def grtl_default(kls, num_nodes, node_features):
        return {"class": kls,
                        "parameters": {
                            "num_nodes": num_nodes,
                            "node_features": node_features
                        }
        }
        