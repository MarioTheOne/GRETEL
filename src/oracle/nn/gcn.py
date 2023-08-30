import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.aggr import MeanAggregation

class GCN(nn.Module):
   
    def __init__(self, node_features, n_classes=2, num_conv_layers=2, num_dense_layers=2, conv_booster=2, linear_decay=2, pooling=MeanAggregation()):
        super(GCN, self).__init__()
        
        self.in_channels = node_features#.shape[-1]
        self.out_channels = int(self.in_channels * conv_booster)
        self.n_classes = n_classes
        self.num_dense_layers = num_dense_layers
        self.linear_decay = linear_decay
        self.pooling = pooling
        
        self.num_conv_layers = [(self.in_channels, self.out_channels)] + [(self.out_channels, self.out_channels) * (num_conv_layers - 1)]
        self.graph_convs = self.__init__conv_layers()
        self.downstream_layers = self.__init__downstream_layers()
        
    def forward(self, node_features, edge_index, edge_weight):
        conv_out = self.graph_convs(node_features, edge_index, edge_weight)
        return self.downstream_layers(conv_out)
    
    def __init__conv_layers(self):
        ############################################
        # initialize the convolutional layers interleaved with pooling layers
        graph_convs = []
        for i in range(len(self.num_conv_layers)):#add len
            graph_convs.append(GCNConv(in_channels=self.num_conv_layers[i][0],
                                      out_channels=self.num_conv_layers[i][1]))
            graph_convs.append(self.pooling)
        # put the conv layers in sequential
        return nn.Sequential(*graph_convs)
    
    def __init__downstream_layers(self):
        ############################################
        # initialize the linear layers interleaved with activation functions
        downstream_layers = []
        in_linear = self.out_channels
        for _ in range(self.num_dense_layers-1):
            downstream_layers.append(nn.Linear(in_linear, in_linear // self.linear_decay))
            downstream_layers.append(nn.ReLU())
            in_linear = in_linear // self.linear_decay
        # add the output layer
        downstream_layers.append(nn.Linear(in_linear, self.n_classes))
        downstream_layers.append(nn.Softmax())
        # put the linear layers in sequential
        return nn.Sequential(*downstream_layers)