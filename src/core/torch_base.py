import torch

from src.core.trainable_base import Trainable
from src.utils.utils import get_instance_kvargs, config_default
from abc import ABCMeta, abstractmethod

class TorchBase(Trainable, metaclass=ABCMeta):
       
    def init(self):
        self.epochs = self.local_config['parameters']['epochs']
        
        self.model = get_instance_kvargs(self.local_config['parameters']['model']['class'],
                                   self.local_config['parameters']['model']['parameters'])

        self.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})
        
        self.loss_fn = get_instance_kvargs(self.local_config['parameters']['loss_fn']['class'],
                                        self.local_config['parameters']['loss_fn']['parameters'])
        
        self.batch_size = self.local_config['parameters']['batch_size']
        
        #TODO: Need to fix GPU support!!!!
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)                            
    
    @abstractmethod     
    def real_fit(self):
        pass
            
    def check_configuration(self, local_config):
        local_config['parameters'] = local_config.get('parameters', {})
        # set defaults
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 100)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 8)
        # populate the optimizer
        config_default(local_config, 'optimizer', 'torch.optim.Adam')
        config_default(local_config, 'loss_fn', 'torch.nn.CrossEntropyLoss')
        return local_config
