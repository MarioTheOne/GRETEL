import copy
from typing import Any, Tuple

import numpy as np
import torch
from torch_geometric.utils.unbatch import unbatch, unbatch_edge_index
from src.n_dataset.utils.dataset_torch import TorchGeometricDataset

from src.core.factory_base import get_instance_kvargs
from src.core.torch_base import TorchBase
from src.n_dataset.instances.graph import GraphInstance
from src.utils.cfg_utils import init_dflts_to_of, retake_oracle
from src.utils.torch.utils import rebuild_adj_matrix


class BaseGAN(TorchBase):
    
    def init(self):
        self.oracle = retake_oracle(self.local_config)
        #We override the init of TorchBase
        local_params = self.local_config['parameters']
        self.epochs = local_params['epochs']
        self.batch_size = local_params['batch_size']
        self.explainee_label = local_params['model_label']       

        # Initialise the generator and its optimizer
        self.generator = get_instance_kvargs(local_params['generator']['class'],
                                             local_params['generator']['parameters'])
        
        self.generator_optimizer = get_instance_kvargs(local_params['gen_optimizer']['class'],
                                             {'params':self.generator.parameters(), 
                                              **local_params['gen_optimizer']['parameters']})  
         
        # Initialise the discriminator and its optimizer
        self.discriminator = get_instance_kvargs(local_params['discriminator']['class'],
                                                 local_params['discriminator']['parameters'])

        self.discriminator_optimizer = get_instance_kvargs(local_params['disc_optimizer']['class'],
                                             {'params':self.discriminator.parameters(), 
                                              **local_params['disc_optimizer']['parameters']})
                
        self.loss_fn = get_instance_kvargs(local_params['loss_fn']['class'],
                                           local_params['loss_fn']['parameters'])
        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.generator.device = self.device
        self.discriminator.device = self.device

        self.model = [
            self.generator,
            self.discriminator
        ]
        
    def infinite_data_stream(self, loader):
        # Define a generator function that yields batches of data
        while True:
            for batch in loader:
                yield batch.to(self.device)
                
    def prepare_discriminator_for_training(self):
        self.discriminator_optimizer.zero_grad()
        self.generator.set_training(False)
        self.generator.train(False)
        self.discriminator.set_training(True)
        self.discriminator.train(True)
        
        
    def prepare_generator_for_training(self):
        self.generator_optimizer.zero_grad() 
        self.generator.set_training(True)
        self.generator.train(True)
        self.discriminator.set_training(False)
        self.discriminator.train(False)
        
    def take_oracle_predictions(self, instances, y_true):
        oracle_scores = [self.oracle.predict_proba(inst)[1-self.explainee_label] for inst in instances]
        # The following update to the oracle scores is needed to have
        # the same order of magnitude between real and generated sample losses
        oracle_scores = np.array(oracle_scores, dtype=float).squeeze()
        real_samples = torch.where(y_true == 1.)
        average_score_real_samples = np.mean(oracle_scores[real_samples])
        if average_score_real_samples != 0:
            oracle_scores[real_samples] /= average_score_real_samples
        
        fake_samples = torch.where(y_true == 0.)
        oracle_scores[fake_samples] = 1.
        oracle_scores = torch.tensor(oracle_scores, dtype=torch.float).to(self.device)
        
        return oracle_scores
        
    def get_generator_kls(self):
        return self._generator_kls
        
    def set_generator_kls(self, kls):
        self._generator_kls = kls
  
    def get_discriminator_kls(self):
        return self._discriminator_kls
        
    def set_discriminator_kls(self, kls):
        self._discriminator_kls = kls
        
    def check_configuration(self):
        # We let TorchBase do some check for us.
        super().check_configuration()
        local_config = self.local_config
        #local_config['parameters']['epochs'] = 2000
        local_config['parameters']['batch_size'] = 1
                
        #Declare the default classes to use
        gen_kls = self.get_generator_kls()
        disc_kls = self.get_discriminator_kls()
        
        #Check if the generator exist or build with its defaults:
        init_dflts_to_of(local_config, 'generator', gen_kls, self.dataset.num_node_features())
        #Check if the generator exist or build with its defaults:
        init_dflts_to_of(local_config, 'discriminator', disc_kls, \
                         self.dataset.num_nodes, self.dataset.num_node_features())
        
        # epochs, batch_size,  optimizer, and the loss_fn are already setted by the TorchBase
        # We do not need the default (generated by the TorchBase) optimizer so we rid of it
        local_config['parameters'].pop('optimizer')

        # We look if it is present a proto_optimizer to use
        if 'proto_optimizer' not in local_config['parameters']:
            init_dflts_to_of(local_config, 'proto_optimizer','torch.optim.SGD',lr=0.001)
        
        proto_optimizer = local_config['parameters'].pop('proto_optimizer')        

        # If the gen_optimizer is not present we create it
        if 'gen_optimizer' not in local_config['parameters']:
            local_config['parameters']['gen_optimizer'] = copy.deepcopy(proto_optimizer)
            local_config['parameters']['gen_optimizer']['parameters']['lr']=0.001 # We  override the proto lr

        # If the gen_optimizer is not present we create it
        if 'disc_optimizer' not in local_config['parameters']:
            local_config['parameters']['disc_optimizer'] = copy.deepcopy(proto_optimizer)
            local_config['parameters']['disc_optimizer']['parameters']['lr']=0.001 # We override the proto lr        
        
        init_dflts_to_of(local_config, 'gen_optimizer','torch.optim.SGD',lr=0.001)
        init_dflts_to_of(local_config, 'disc_optimizer','torch.optim.SGD',lr=0.001)
        
    def __call__(self, *args: Tuple[GraphInstance], **kwds: Any) -> Any:
        batch = TorchGeometricDataset.to_geometric(args[0]).to(self.device)
        return self.generator(batch.x, batch.edge_index, batch.edge_attr, batch.batch)