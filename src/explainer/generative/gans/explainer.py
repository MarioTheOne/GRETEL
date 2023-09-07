
import torch

from src.core.explainer_base import Explainer
from src.core.torch_base import TorchBase
from src.n_dataset.utils.dataset_torch import TorchGeometricDataset
from src.utils.utils import get_instance_kvargs


class GANExplainer(Explainer, TorchBase):
    
    def init(self):        
        self.models = [
            get_instance_kvargs(self.local_config['parameters']['proto_model']['class'],
                                self.local_config['parameters']['proto_model']['parameters'])\
                                    for _ in range(self.dataset.num_classes)
        ]
        
        self.sampler = get_instance_kvargs(self.local_config['parameters']['sampler']['class'],
                                           self.local_config['parameters']['sampler']['parameters'])

    
    def real_fit(self):
        for model in self.models:
            model.read_fit()
            
    def explain(self, instance):            
        with torch.no_grad():
            #######################################################
            batch = TorchGeometricDataset.to_geometric(instance)            
            pred_label = self.oracle.predict(instance)
            embedded_features, _, edge_probs = self.models[pred_label].generator(batch.x, batch.edge_index, batch.edge_attr, None)
            instance = self.sampler.sample(instance, **{'embedded_features': embedded_features,
                                                         'edge_probabilities': edge_probs,
                                                         'edge_index': batch.edge_index})
        return instance
    
    
    def check_configuration(self, local_config):
        #####################################################################
        # TODO: get_default classmethod for all Base objects
        if 'proto_model' not in self.local_config['parameters']:
            local_config['parameters']['proto_model'] = {
                "class": "src.explainer.generative.gans.model.GAN",
                "parameters": {
                    "generator": {
                        "class": "src.explainer.generative.gans.generators.residual_generator.ResGenerator",
                        "parameters": {
                            "node_features": self.dataset.num_node_features()
                        }
                    },
                    "discriminator": {
                        "class": "src.explainer.generative.gans.discriminators.SimpleDiscriminator",
                        "parameters": {
                            "n_nodes": self.dataset.num_nodes,
                            "node_features": self.dataset.num_node_features()
                        }
                    },
                    "lr": {
                        "generator": 0.001,
                        "discriminator": 0.01
                    }
                }
            }
            
        if 'sampler' not in self.local_config['parameters']:
            local_config['parameters']['sampler'] = {
                "class": "src.n_samplers.partial_order_samplers.PositiveAndNegativeEdgeSampler",
                "parameters": {}
            }
        
        #####################################################################
        return local_config
    
    