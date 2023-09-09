
import copy
import numpy as np
import torch

from src.core.torch_base import TorchBase
from src.n_dataset.instances.graph import GraphInstance
from src.utils.cfg_utils import init_dflts_to_of
from src.utils.torch.utils import rebuild_adj_matrix
from src.core.factory_base import get_instance_kvargs


class GAN(TorchBase):
    
    def init(self):
        #We override the init of TorchBase
        local_params = self.local_config['parameters']
        self.epochs = local_params['epochs']
        self.batch_size = local_params['batch_size']

        self.generator = get_instance_kvargs(local_params['generator']['class'],
                                             local_params['generator']['parameters'])
    
        self.discriminator = get_instance_kvargs(local_params['discriminator']['class'],
                                                 local_params['discriminator']['parameters'])    
        
        self.generator_optimizer = get_instance_kvargs(local_params['gen_optimizer']['class'],
                                             {'params':self.generator.parameters(), 
                                              **local_params['gen_optimizer']['parameters']})
        
        self.discriminator_optimizer = get_instance_kvargs(local_params['disc_optimizer']['class'],
                                             {'params':self.discriminator.parameters(), 
                                              **local_params['disc_optimizer']['parameters']})   
        
        self.loss_fn = get_instance_kvargs(local_params['loss_fn']['class'],
                                           local_params['loss_fn']['parameters'])
        
        
    def real_fit(self):
        discriminator_loader = self.dataset.get_torch_loader(fold_id=self.fold_id, kls=self.explainee_label)
        # TODO: make it multiclass in Dataset
        generator_loader = self.dataset.get_torch_loader(fold_id=self.fold_id, kls=1-self.explainee_label)
        
        for epoch in range(self.epochs):
            G_losses, D_losses = [], []

            if epoch > 0:
                encoded_features, _, edge_probs = self.generator(node_features, edge_index, edge_features)
                if self._check_divergence(encoded_features, edge_probs):
                    break
            #######################################################################
            self.discriminator_optimizer.zero_grad()
            self.generator.set_training_generator(False)
            self.generator.train(False)
            self.discriminator.set_training_discriminator(True)
            self.discriminator.train(True)
            #######################################################################
            # discriminator data (real batch)
            node_features, edge_index, edge_features, _ = next(discriminator_loader)
            self._logger.info("Passed Neverending loop.")
            #######################################################################
            # generator data (fake batch)
            fake_node_features, fake_edge_index, fake_edge_features, _ = next(generator_loader)
            _, fake_edge_index, edge_probs = self.generator(fake_node_features, fake_edge_index, fake_edge_features)
            # get the real and fake labels
            y_batch = torch.cat([torch.ones((self.batch_size,)), torch.zeros((self.batch_size,))], dim=0)
            #######################################################################
            # get the oracle's predictions
            oracle_scores = []
            real_inst = GraphInstance(data=rebuild_adj_matrix(len(node_features), edge_index, edge_features).numpy(),
                                 node_features=node_features,
                                 edge_features=edge_features)
            
            fake_inst = GraphInstance(data=rebuild_adj_matrix(len(fake_node_features), fake_edge_index, fake_edge_features).numpy(),
                                      node_features=fake_node_features,
                                      edge_features=fake_edge_features)
            
            oracle_scores = [self.oracle.predict_proba(inst)[1-self.explainee_label] for inst in [real_inst, fake_inst]]
            # The following update to the oracle scores is needed to have
            # the same order of magnitude between real and generated sample losses
            oracle_scores = np.array(oracle_scores, dtype=float).squeeze()
            real_samples = torch.where(y_batch == 1.)
            average_score_real_samples = np.mean(oracle_scores[real_samples])
            if average_score_real_samples != 0:
                oracle_scores[real_samples] /= average_score_real_samples
            
            fake_samples = torch.where(y_batch == 0.)
            oracle_scores[fake_samples] = 1.
            
            #######################################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            real_pred = self.discriminator(node_features, edge_index, edge_features).expand(1)
            fake_pred = self.discriminator(fake_node_features, fake_edge_index, fake_edge_features).expand(1)
            y_pred = torch.cat([real_pred, fake_pred])
            loss = torch.mean(self.loss_fn(y_pred.squeeze().double(), y_batch.double()) * torch.tensor(oracle_scores, dtype=torch.float))
                    
            D_losses.append(loss.item())
            loss.backward()
            self.discriminator_optimizer.step()
            #######################################################################
            self.generator_optimizer.zero_grad() 
            self.generator.set_training_generator(True)
            self.generator.train(True)
            self.discriminator.set_training_discriminator(False)
            self.discriminator.train(False)
            #######################################################################
            ## Update G network: maximize log(D(G(z)))
            # generator data (fake batch)
            fake_node_features, fake_edge_index, fake_edge_features, _ = next(generator_loader)
            y_fake = torch.ones((self.batch_size,))
            output = self.discriminator(self.generator(fake_node_features, fake_edge_index, fake_edge_features), fake_edge_index, fake_edge_features)
            # calculate the loss
            loss = self.loss_fn(output.expand(1).double(), y_fake.double())
            loss.backward()
            G_losses.append(loss.item())
            self.generator_optimizer.step()
                
            self._logger.info(f'Epoch {epoch}\t Loss_D = {np.mean(D_losses): .4f}\t Loss_G = {np.mean(G_losses): .4f}')
        
    def _check_divergence(self, generated_features, generated_edge_probs):
      return torch.all(torch.isnan(generated_features)) or torch.all(torch.isnan(generated_edge_probs))
  
    def check_configuration(self):
        # We let TorchBase do some check for us.
        super().check_configuration()
        local_config = self.local_config        
        
        #Declare the default classes to use
        gen_kls='src.explainer.generative.gans.res_gen.ResGenerator'
        disc_kls='src.explainer.generative.gans.smpl_disc.SimpleDiscriminator'  

        #Check if the generator exist or build with its defaults:
        init_dflts_to_of(local_config, 'generator', gen_kls, self.dataset.num_node_features())
        #Check if the generator exist or build with its defaults:
        init_dflts_to_of(local_config, 'discriminator', disc_kls, \
                         self.dataset.num_nodes, self.dataset.num_node_features())
        
        # epochs, batch_size,  optimizer, and the loss_fn are already setted by the TorchBase
        # We assume that the optimizer if it is user setted or not is used as prototype
        proto_optimizer = local_config['parameters'].pop('optimizer')

        init_dflts_to_of(local_config, 'gen_optimizer','torch.optim.SGD')
        init_dflts_to_of(local_config, 'disc_optimizer','torch.optim.SGD')

        # If the gen_optimizer is not present we create it
        if 'gen_optimizer' not in local_config['parameters']:
            local_config['parameters']['gen_optimizer'] = copy.deepcopy(proto_optimizer)
            #local_config['parameters']['gen_optimizer']['parameters']['lr']=0.001 # We opt to override the proto lr
        else:
            init_dflts_to_of(local_config, 'gen_optimizer','torch.optim.Adam') # We opt to not override user's lr
        
        if 'disc_optimizer' not in local_config['parameters']:
            local_config['parameters']['disc_optimizer'] = copy.deepcopy(proto_optimizer)
            #local_config['parameters']['disc_optimizer']['parameters']['lr']=0.01 # We opt to override the proto lr
        else:
            init_dflts_to_of(local_config, 'disc_optimizer','torch.optim.Adam') # We opt to not override user's lr
        

   
        
        
