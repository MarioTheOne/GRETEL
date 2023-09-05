from src.core.embedder_base import Embedder
from src.dataset.dataset_base import Dataset

import numpy as np
from rdkit.Chem import RDKFingerprint


class RDKFingerprintEmbedder(Embedder):

    def init(self):
        self.model = []

    def real_fit(self):
        for inst in self.dataset.instances:
            fingerprint = RDKFingerprint(inst.data)
            fingerprint_np = np.array(fingerprint)
            self.model.append(fingerprint_np)

    def get_embeddings(self):
        return self.model

    def get_embedding(self, instance):
        fingerprint_rdk = RDKFingerprint(instance.data)
        fingerprint_rdk_np = np.array(fingerprint_rdk)
        return fingerprint_rdk_np
    
    
    
