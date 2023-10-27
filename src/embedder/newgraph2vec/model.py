import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from src.core.embedder_base import Embedder
from src.embedder.newgraph2vec.graph2vec_model import WeisfeilerLehmanMachine


class Graph2VecEmbedder(Embedder):

    def get_embeddings(self):
        return np.array(self._embedding)

    def get_embedding(self, instance):
        graph = instance.get_nx()
        #todo get features
        document = WeisfeilerLehmanMachine(graph, None, self.wl_iterations)
        document = TaggedDocument(words=document.extracted_features, tags=[str(0)])
        return np.array(self.model.infer_vector(document.words))

    def real_fit(self):
        graphs = [g.get_nx() for g in self.dataset.instances]
        #todo get or set features for g
        documents = [
            WeisfeilerLehmanMachine(
                graph, None, self.wl_iterations
            )
            for graph in graphs
        ]
        documents = [
            TaggedDocument(words=doc.extracted_features, tags=[str(i)])
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

        self._embedding = [self.model.docvecs[str(i)] for i, _ in enumerate(documents)]

    def check_configuration(self):
        super().check_configuration()
        # self.window
        # self.min_count
        # self.sample
        # self.workers
        # self.epochs
        # self.alpha
        # self.seed