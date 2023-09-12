
import copy
from types import NoneType

import numpy as np
import torch

from src.core.factory_base import get_instance_kvargs
from src.core.torch_base import TorchBase
from src.n_dataset.instances.graph import GraphInstance
from src.n_dataset.utils.dataset_torch import TorchGeometricDataset
from src.utils.cfg_utils import init_dflts_to_of, retake_oracle
from src.utils.torch.utils import rebuild_adj_matrix


class GAN(TorchBase):
    
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
        
        self.model = [
            self.generator,
            self.discriminator
        ]
        
    def __infinite_data_stream(self, loader):
        # Define a generator function that yields batches of data
        while True:
            for batch in loader:
                yield batch
                
    def real_fit(self):
        discriminator_loader = self.__infinite_data_stream(self.dataset.get_torch_loader(fold_id=self.fold_id, kls=self.explainee_label))
        # TODO: make it multiclass in Datase
        generator_loader = self.__infinite_data_stream(self.dataset.get_torch_loader(fold_id=self.fold_id, kls=1-self.explainee_label))

        for epoch in range(self.epochs):
            G_losses, D_losses = [], []
            #######################################################################
            self.prepare_discriminator_for_training()
            #######################################################################
            # discriminator data (real batch)
            node_features, edge_index, edge_features, _ , real_batch ,_ = next(discriminator_loader)
            # generator data (fake batch)
            fake_node_features, fake_edge_index, fake_edge_features, _ , fake_batch , _  = next(generator_loader)
            fake_node_features, fake_edge_index, fake_edge_features = self.generator(fake_node_features[1], fake_edge_index[1], fake_edge_features[1], fake_batch[1])
            # get the real and fake labels
            y_batch = torch.cat([torch.ones((len(torch.unique(real_batch[1])),)),
                                 torch.zeros(len(torch.unique(fake_batch[1])),)], dim=0)
            #######################################################################
            # get the oracle's predictions            
            real_inst = self.__retake_batch(node_features[1], edge_index[1], edge_features[1], real_batch[1])
            fake_inst = self.__retake_batch(fake_node_features, fake_edge_index, fake_edge_features, fake_batch[1], counterfactual=True, generator=True)
            
            oracle_scores = self.take_oracle_predictions(real_inst + fake_inst, y_batch)
            #######################################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            loss, _ = self.optimize_discriminator(real_inst + fake_inst, y_batch, oracle_scores)                    
            D_losses.append(loss.item())
            loss.backward()
            self.discriminator_optimizer.step()
            #######################################################################
            self.prepare_generator_for_training()
            ## Update G network: maximize log(D(G(z)))
            fake_node_features, fake_edge_index, fake_edge_features,  _ , fake_batch ,_ = next(generator_loader)
            loss = self.optimize_generator(fake_node_features[1], fake_edge_index[1], fake_edge_features[1], fake_batch[1])
            loss.backward()
            G_losses.append(loss.item())
            self.generator_optimizer.step()
                
            self.context.logger.info(f'Epoch {epoch}\t Loss_D = {np.mean(D_losses): .4f}\t Loss_G = {np.mean(G_losses): .4f}')
  
  
    def __retake_batch(self, node_features, edge_indices, edge_features, batch, counterfactual=False, generator=False):
        instances = []
        unique_batch_indices = torch.unique(batch)
        for index in unique_batch_indices:
            batch_indices = torch.where(batch == index)[0]
            batch_edge_indices = edge_indices[:,batch_indices]
            batch_edge_features = edge_features[batch_indices]
            if generator:
                batch_edge_features = edge_features[batch_edge_indices[0,:], batch_edge_indices[1,:]]
            instances.append(GraphInstance(id=0,
                                           label=1-self.explainee_label if counterfactual else self.explainee_label,
                                           data=rebuild_adj_matrix(len(node_features[batch_indices]), batch_edge_indices, batch_edge_features.T).detach().numpy(),
                                           node_features=node_features[batch_indices].detach().numpy(),
                                           edge_features=batch_edge_features.detach().numpy()))
        return instances
    
    def optimize_discriminator(self, instances, y_true, oracle_scores):
        preds = torch.zeros((len(instances), self.dataset.num_classes))
        for i, inst in enumerate(instances):
            data = TorchGeometricDataset.to_geometric(inst)
            node_features, edge_index, edge_features = data.x, data.edge_index, data.edge_attr
            y_pred = self.discriminator(node_features, edge_index, edge_features)
            preds[i] = y_pred
        loss = torch.mean(self.loss_fn(preds.squeeze().double(), y_true.long()) * oracle_scores)
        return loss, preds
    
    
    def optimize_generator(self, node_features, edge_index, edge_features, batch):
        fake_node_features, fake_edge_index, fake_edge_features = self.generator(node_features, edge_index, edge_features, batch)
        instances = self.__retake_batch(fake_node_features, fake_edge_index, fake_edge_features, batch, generator=True)
        y_fake = torch.ones((len(torch.unique(batch)),))
        loss, _ = self.optimize_discriminator(instances, y_fake, torch.ones((len(y_fake),)))
        return loss
    
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
        oracle_scores = torch.tensor(oracle_scores, dtype=torch.float)
        
        return oracle_scores
        
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
            local_config['parameters']['disc_optimizer']['parameters']['lr']=0.01 # We override the proto lr        
        
        init_dflts_to_of(local_config, 'gen_optimizer','torch.optim.SGD',lr=0.001)
        init_dflts_to_of(local_config, 'disc_optimizer','torch.optim.SGD',lr=0.01)

   
        
        
