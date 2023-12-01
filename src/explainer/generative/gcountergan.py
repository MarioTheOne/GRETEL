import torch
import copy
from src.core.factory_base import get_instance_kvargs
from src.explainer.per_cls_explainer import PerClassExplainer

from src.n_dataset.utils.dataset_torch import TorchGeometricDataset
from src.utils.cfg_utils import init_dflts_to_of

class GCounteRGAN(PerClassExplainer):


    def init(self):
        super().init()
        self.sampler = get_instance_kvargs(self.local_config['parameters']['sampler']['class'],
                                        self.local_config['parameters']['sampler']['parameters'])

    def explain(self, instance):            
        with torch.no_grad():
            #######################################################
            batch = TorchGeometricDataset.to_geometric(instance).to(self.device)
            embedded_features, edge_probs = dict(), dict()
            for i, explainer in enumerate(self.model):
                node_features, _, probs = explainer.generator(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                embedded_features[i] = node_features
                edge_probs[i] = probs.cpu().numpy()
            
            cf_instance = self.sampler.sample(instance, self.oracle, **{'embedded_features': embedded_features,
                                                                        'edge_probabilities': edge_probs})            
        return cf_instance if cf_instance else instance
    
    def check_configuration(self):
        super().check_configuration()
        #The sampler must be present in any case
        init_dflts_to_of(self.local_config,
                         'sampler',
                         'src.utils.n_samplers.partial_order_samplers.Bernouilli',
                         sampling_iterations=500)
    