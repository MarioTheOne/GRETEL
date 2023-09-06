
import torch

from src.core.torch_base import TorchBase
from src.utils.utils import get_instance_kvargs


class GAN(TorchBase):
    
    def init(self):
        self.generator = get_instance_kvargs(self.local_config['parameters']['generator']['class'],
                                             self.local_config['parameters']['generator']['parameters'])
    
        self.discriminator = get_instance_kvargs(self.local_config['parameters']['discriminator']['class'],
                                                 self.local_config['parameters']['discriminator']['parameters'])
        
        self.explainee_label = self.local_config['parameters']['explainee_label']
        
    def real_fit(self):
        discriminator_loader = self.dataset.get_torch_loader(fold_id=self.fold_id, kls=self.explainee_label)
        # TODO: make it multiclass in Dataset
        generator_loader = self.dataset.get_torch_loader(fold_id=self.fold_id, kls=1-self.explainee_label)
        
        
    def _check_divergence(self, generated_features: torch.Tensor, generated_edge_probs: torch.Tensor):
      return torch.all(torch.isnan(generated_features)) or torch.all(torch.isnan(generated_edge_probs))
        