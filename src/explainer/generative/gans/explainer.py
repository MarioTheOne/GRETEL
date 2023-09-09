
import torch

from src.core.explainer_base import Explainer
from src.core.factory_base import get_instance_kvargs
from src.core.trainable_base import Trainable
from src.n_dataset.utils.dataset_torch import TorchGeometricDataset
from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, set_if_not


class GANExplainer(Explainer, Trainable):
    
    def init(self):
        # TODO I don't know why this needs to propagate. Check it
        # CHECK ASAP
        self.local_config['dataset'] = self.dataset
        self.local_config['parameters']['proto_model']['parameters']['dataset'] = self.dataset
        self.local_config['parameters']['proto_model']['parameters']['parameters'] = {'fold_id': self.local_config['parameters']['fold_id']}
        
        self.models = [
            get_instance_kvargs(self.local_config['parameters']['proto_model']['class'],
                                {
                                    "context": self.context, 
                                    "local_config": self.local_config['parameters']['proto_model']['parameters'],
                                    **self.local_config['parameters']['proto_model']['parameters'],
                                }) for _ in range(self.dataset.num_classes)
        ]
        # TODO: check how to pass this more elegantly
        for i, model in enumerate(self.models):
            model.explainee_label = i
                
        self.sampler = get_instance_kvargs(self.local_config['parameters']['sampler']['class'],
                                           self.local_config['parameters']['sampler']['parameters'])

    
    def real_fit(self):
        for model in self.models:
            model.real_fit()
            
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
        super.check_configuration(local_config)
        
        proto_kls='src.explainer.generative.gans.model.GAN'
        gen_kls='src.explainer.generative.gans.generators.residual_generator.ResGenerator'
        disc_kls='src.explainer.generative.gans.discriminators.simple_discriminator.SimpleDiscriminator'

        get_dflts_to_of(local_config, 'proto_model',proto_kls)
        proto_model_snippet = local_config['parameters']['proto_model']

        init_dflts_to_of(proto_model_snippet, 'generator', gen_kls, self.dataset.num_node_features())
        init_dflts_to_of(proto_model_snippet, 'discriminator', disc_kls, \
                         self.dataset.num_nodes,self.dataset.num_node_features())
        
        set_if_not(proto_model_snippet, 'lr', {'generator': 0.001,'discriminator': 0.01})
                   
        init_dflts_to_of(local_config,'sampler','src.utils.n_samplers.partial_order_samplers.PositiveAndNegativeEdgeSampler')

        #TODO: those are useles call if are equal to defaults of TorchBase
        init_dflts_to_of(proto_model_snippet, 'optimizer','torch.optim.Adam') 
        init_dflts_to_of(proto_model_snippet, 'loss_fn','torch.nn.CrossEntropyLoss')
        return local_config
    
    