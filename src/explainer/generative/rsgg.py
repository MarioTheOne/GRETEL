
import copy
import torch
from src.core.explainer_base import Explainer
from src.core.factory_base import get_instance_kvargs
from src.core.trainable_base import Trainable
from src.n_dataset.utils.dataset_torch import TorchGeometricDataset
from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, inject_dataset, inject_oracle

class RSGG(Trainable, Explainer):
    
    
    def init(self):       
        self.model = [ get_instance_kvargs(model['class'],
                    {'context':self.context,'local_config':model}) for model in self.local_config['parameters']['models']]
                       
        self.sampler = get_instance_kvargs(self.local_config['parameters']['sampler']['class'],
                                           self.local_config['parameters']['sampler']['parameters'])
        
    def real_fit(self):
        pass # Not neede because the GAN are trained by their constructor
        '''for model in self.models:
            model.real_fit()'''
        
    def write(self):
        pass
            
    def explain(self, instance):            
        with torch.no_grad():
            #######################################################
            batch = TorchGeometricDataset.to_geometric(instance)            
            pred_label = self.oracle.predict(instance)
            _, _, edge_probs = self.model[pred_label].generator(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            cf_instance = self.sampler.sample(instance, self.oracle, **{'node_features': batch.x,
                                                                     'edge_probabilities': edge_probs,
                                                                     'edge_index': batch.edge_index})
        return cf_instance if cf_instance else instance
    
    def check_configuration(self):
        super().check_configuration()
        
        proto_kls='src.explainer.generative.gans.model.GAN'

        #The sampler must be present in any case
        #TODO: bug sul sampling iteration
        init_dflts_to_of(self.local_config,'sampler','src.utils.n_samplers.partial_order_samplers.PositiveAndNegativeEdgeSampler', sampling_iterations=500)
        
        # Check if models is present and of the right size
        if 'models' not in self.local_config['parameters'] or len(self.local_config['parameters']['models']) < self.dataset.num_classes:
            #Check if the models exist or create it
            if 'models' not in self.local_config['parameters']:
                self.local_config['parameters']['models']=[]

            cfg_models = self.local_config['parameters']['models']
            if 'proto_model' not in self.local_config['parameters'] and len(cfg_models)==1:
                # Assume that the only passed model can be used as prototype for all
                proto_snippet = copy.deepcopy(cfg_models[0])
            else:
                # We need to get and check the proto or create it if not available
                get_dflts_to_of(self.local_config, 'proto_model',proto_kls)
                proto_snippet = self.local_config['parameters'].pop('proto_model')

            # Add the missed per class models
            models = []
            for i in range(self.dataset.num_classes):
                model=copy.deepcopy(proto_snippet)
                model['parameters']['model_label'] = i

                for usr_model in cfg_models:
                    if(usr_model['parameters']['model_label'] == i):
                        # We sobstitute the copied prototype
                        model = usr_model
                
                # In any case we need to inject oracle and the dataset to the model
                inject_dataset(model, self.dataset)
                inject_oracle(model, self.oracle)

                # Check if the fold_id is present is inherited otherwise
                model['parameters']['fold_id'] = model['parameters'].get('fold_id',self.fold_id)

                #Propagate teh retrain
                retrain = self.local_config['parameters'].get('retrain', False)
                model['parameters']['retrain']=retrain

                models.append(model)

            # We replace models with the checked ones
            self.local_config['parameters']['models']=models