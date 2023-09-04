from src.core.oracle_base import Oracle
from src.utils.utils import add_init_defaults_params

class TabularOracle(Oracle):   
    def init(self):
        embedding_snippet = self.local_config['parameters']['embedder'] 
        self.embedder = self.context.factories['embedders'].get_embedder(embedding_snippet)
        
    def real_fit(self):
        fold_id = self.local_config['parameters']['fold_id']
        inst_vectors = self.embedder.get_embeddings()

        if fold_id == -1:
            # Training with the entire dataset
            x = inst_vectors
            y = [ i.label for i in self.dataset.instances]
        else:
            # Training with an specific split
            spt = self.dataset.get_split_indices()[fold_id]['train']
            x = [inst_vectors[i] for i in spt]
            y = [self.dataset.get_instance(i).label for i in spt]
        self.model.fit(x, y)

    def _real_predict(self, data_instance):
        data_instance = self.embedder.get_embedding(data_instance)
        return self.model.predict(data_instance)
        
    def _real_predict_proba(self, data_instance):
        data_instance = self.embedder.get_embedding(data_instance)
        return self.model.predict_proba(data_instance).squeeze()
    
    def check_configuration(self, local_config):
        kls = "src.embedder.graph2vec.model.Graph2VecEmbedder"
        if 'embedder' not in local_config['parameters']:
            local_config['parameters']['embedder'] = {
                "class": kls, 
                "parameters": {
                    "model": {
                        "parameters": { 
                            "wl_iterations": 2
                        }
                    }
                }
            }
        node_config = add_init_defaults_params(kls, local_config['parameters']['embedder']['parameters']['model']['parameters'])
        local_config['parameters']['embedder']['parameters']['model']['parameters'] = node_config
        # populate the local config accordingly
        local_config['parameters']['embedder']['dataset'] = self.dataset
        local_config['parameters']['embedder']['parameters']['fold_id'] = local_config['parameters']['fold_id']
        return local_config
    
    