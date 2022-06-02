from src.oracle.embedder_base import Embedder
from src.dataset.dataset_base import Dataset

import numpy as np
import networkx as nx
from typing import List
from karateclub.estimator import Estimator
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing


class Graph2vec(Embedder):

    def __init__(self, id) -> None:
        super().__init__(id)
        self.emb = custom_g2v()
        self._name = 'graph2vec'

    def fit(self, dataset : Dataset):
        graphs = [i.graph for i in dataset.get_data()]
        self.emb.fit(graphs)

    def get_embeddings(self):
        return self.emb.get_embedding()

    def get_embedding(self, instance):
        return self.emb.infer(instance.graph)


class custom_g2v(Estimator):
    """An implementation of `"Graph2Vec" <https://arxiv.org/abs/1707.05005>`_
    from the MLGWorkshop '17 paper "Graph2Vec: Learning Distributed Representations of Graphs".
    The procedure creates Weisfeiler-Lehman tree features for nodes in graphs. Using
    these features a document (graph) - feature co-occurence matrix is decomposed in order
    to generate representations for the graphs.

    The procedure assumes that nodes have no string feature present and the WL-hashing
    defaults to the degree centrality. However, if a node feature with the key "feature"
    is supported for the nodes the feature extraction happens based on the values of this key.

    Args:
        wl_iterations (int): Number of Weisfeiler-Lehman iterations. Default is 2.
        attributed (bool): Presence of graph attributes. Default is False.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        down_sampling (float): Down sampling frequency. Default is 0.0001.
        epochs (int): Number of epochs. Default is 10.
        learning_rate (float): HogWild! learning rate. Default is 0.025.
        min_count (int): Minimal count of graph feature occurrences. Default is 5.
        seed (int): Random seed for the model. Default is 42.
        erase_base_features (bool): Erasing the base features. Default is False.
    """

    def __init__(
        self,
        wl_iterations: int = 2,
        attributed: bool = False,
        dimensions: int = 128,
        workers: int = 4,
        down_sampling: float = 0.0001,
        epochs: int = 10,
        learning_rate: float = 0.025,
        min_count: int = 5,
        seed: int = 42,
        erase_base_features: bool = False,
    ):

        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.dimensions = dimensions
        self.workers = workers
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self.erase_base_features = erase_base_features

 
    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)


    def fit(self, graphs: List[nx.classes.graph.Graph]):
        """
        Fitting a Graph2Vec model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        graphs = self._check_graphs(graphs)
        documents = [
            WeisfeilerLehmanHashing(
                graph, self.wl_iterations, self.attributed, self.erase_base_features
            )
            for graph in graphs
        ]
        documents = [
            TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
            for i, doc in enumerate(documents)
        ]

        self.model = Doc2Vec(
            documents,
            vector_size=self.dimensions,
            window=0,
            min_count=self.min_count,
            dm=0,
            sample=self.down_sampling,
            workers=self.workers,
            epochs=self.epochs,
            alpha=self.learning_rate,
            seed=self.seed,
        )

        # self.model = Doc2Vec(vector_size=self.dimensions, alpha=self.learning_rate,
        #                  min_alpha=self.learning_rate, min_count=1, dm=0, seed=self.seed)  # use fixed learning rate
        # self.model.build_vocab(documents)
        # for epoch in range(10):
        #     self.model.train(documents, total_examples=self.model.corpus_count, epochs=self.epochs)
        #     self.model.alpha -= 0.002  # decrease the learning rate
        #     self.model.min_alpha = self.model.alpha  # fix the learning rate, no decay

        self._embedding = [self.model.docvecs[str(i)] for i, _ in enumerate(documents)]

    
    def infer(self, graph: nx.classes.graph.Graph):
        """
        Fitting a Graph2Vec model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        graph = self._check_graphs([graph])[0]
        document = WeisfeilerLehmanHashing(graph, self.wl_iterations, self.attributed, self.erase_base_features)
        document = TaggedDocument(words=document.get_graph_features(), tags=[str(0)])

        self.model.random.seed(self.seed) # Force the seed to eliminate randomness
        return np.array(self.model.infer_vector(document.words))

        