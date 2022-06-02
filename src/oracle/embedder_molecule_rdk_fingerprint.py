from src.oracle.embedder_base import Embedder
from src.dataset.dataset_base import Dataset

import numpy as np
import networkx as nx
from typing import List
from karateclub.estimator import Estimator
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import rdMolDescriptors


class RDKFingerprintEmbedder(Embedder):

    def __init__(self, id) -> None:
        super().__init__(id)
        self._name = 'RDKFingerprint'
        self._embbedings = []

    def fit(self, dataset : Dataset):
        for inst in dataset.instances:
            fingerprint = RDKFingerprint(inst.molecule)
            fingerprint_np = np.array(fingerprint)
            self._embbedings.append(fingerprint_np)

    def get_embeddings(self):
        return self._embbedings

    def get_embedding(self, instance):
        fingerprint_rdk = RDKFingerprint(instance.molecule)
        fingerprint_rdk_np = np.array(fingerprint_rdk)
        return fingerprint_rdk_np
