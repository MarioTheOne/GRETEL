import networkx as nx
import numpy as np

from src.dataset.converters.weights_converter import \
    DefaultFeatureAndWeightConverter
from src.dataset.data_instance_base import DataInstance
from src.dataset.data_instance_features import DataInstanceWFeatures, DataInstanceWFeaturesAndWeights

"""
    Adaptation of Tree-Cycles conversion based on the
    graph classification converter from
    the original repository
"""
class TreeCyclesConverter(DefaultFeatureAndWeightConverter):
    
    def __init__(self, feature_dim=10):
        super(TreeCyclesConverter, self).__init__()
        self.name = 'tree_cycles_converter'
        self.feature_dim = feature_dim
        
    def convert_instance(self, instance: DataInstance) -> DataInstanceWFeaturesAndWeights:
        converted_instance = super().convert_instance(instance)
        weights, features, adj_matrix = self.__preprocess(converted_instance)
        converted_instance.weights = weights
        converted_instance.features = features
        converted_instance.from_numpy_array(adj_matrix)
        return converted_instance
        
    def __preprocess(self, instance: DataInstanceWFeaturesAndWeights) -> np.ndarray:
        graph = self.__sort_nodes_by_degree(instance.graph)
        instance.from_numpy_matrix(nx.adjacency_matrix(graph))
        instance = self.__generate_node_features(instance)
        # set the node features
        adj = instance.to_numpy_array()
        # define the new adjacency matrix which is a full one matrix
        new_adj = np.where(adj != 0, 1, 0)
        # the weights need to be an array of real numbers with
        # length equal to the number of edges
        row_indices, col_indices = np.where(adj != 0)
        weights = adj[row_indices, col_indices]
      
        return weights, instance.features, new_adj
    
    
    def __sort_nodes_by_degree(self, original_graph: nx.Graph) -> nx.Graph:
        # Step 1: Compute the degrees of all nodes
        degrees = dict(original_graph.degree())
        # Step 2: Sort the nodes based on their degrees in descending order
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
        # Step 3: Create a new empty graph with the same attributes as the original graph
        new_graph = nx.Graph(original_graph.graph)
        # Step 4: Iterate over the sorted nodes and add them to the new graph
        for node in sorted_nodes:
            new_graph.add_node(node)
        # Step 5: Add the edges and edge weights from the original graph to the new graph
        for u, v, data in original_graph.edges(data=True):
            if u in new_graph and v in new_graph:
                new_graph.add_edge(u, v, **data)
                
        return new_graph
    
    
    def __generate_node_features(self, instance: DataInstance) -> DataInstanceWFeatures:
        graph = instance.graph
        # Calculate the degree of each node
        degree = dict(graph.degree())
        # Calculate the betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(graph)
        # Calculate the closeness centrality
        closeness_centrality = nx.closeness_centrality(graph)
        # Calculate the harmonic centrality
        harmonic_centrality = nx.harmonic_centrality(graph)
        # Calculate the clustering coefficient
        clustering_coefficient = nx.clustering(graph)
        # Calculate the Katz centrality
        katz_centrality = nx.katz_centrality(graph)
        # Calculate the second order centrality
        second_order_centrality = nx.second_order_centrality(graph)
        # Calculate the Laplacian centrality
        laplacian_centrality = nx.laplacian_spectrum(graph)
        # stack the above calculations and transpose the matrix
        # the new dimensionality is num_nodes x 4
        features = np.stack((list(degree.values()),
                             list(betweenness_centrality.values()),
                             list(closeness_centrality.values()),
                             list(harmonic_centrality.values()),
                             list(clustering_coefficient.values()),
                             list(katz_centrality.values()),
                             list(second_order_centrality.values()),
                             list(laplacian_centrality)), axis=0).T
        # copy the instance information and set the node features
        new_instance = DataInstanceWFeatures(id=instance.id)
        new_instance.from_numpy_matrix(nx.adjacency_matrix(graph))
        new_instance.features = features
        new_instance.graph_label = instance.graph_label
        new_instance.graph_dgl = instance.graph_dgl
        new_instance.edge_labels = instance.edge_labels
        new_instance.node_labels = instance.node_labels
        new_instance.name = instance.name
        # return the new instance with features
        return new_instance