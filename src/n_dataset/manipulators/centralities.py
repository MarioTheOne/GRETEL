from src.n_dataset.manipulators.base import BaseManipulator

import numpy as np
import networkx as nx
class NodeCentrality(BaseManipulator):
    
    
    def node_info(self, instance):
        graph = instance._nx_repr
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
        katz_centrality = nx.katz_centrality_numpy(graph)
        # Calculate the second order centrality
        second_order_centrality = nx.second_order_centrality(graph) if nx.is_connected(graph) else self.__centrality_unconnected_graphs(graph)
        # Calculate the Laplacian centrality
        laplacian_centrality = nx.laplacian_spectrum(graph)
        # feature dictionary
        feature_map = {
            "degrees": list(degree.values()),
            "betweenness": list(betweenness_centrality.values()),
            "closeness": list(closeness_centrality.values()),
            "harmonic_centrality": list(harmonic_centrality.values()),
            "clustering_coefficient": list(clustering_coefficient.values()),
            "katz_centrality": list(katz_centrality.values()),
            "second_order_centrality": list(second_order_centrality.values()),
            "laplacian_centrality": list(laplacian_centrality)
        }
        return feature_map
    
    def __centrality_unconnected_graphs(self, G):
        result = {}
        connected_components = list(nx.connected_components(G))

        for cn in connected_components:
            so_cen = nx.second_order_centrality(G.subgraph(cn))

            for k, v in so_cen.items():
                result[k] = v

        return result
    