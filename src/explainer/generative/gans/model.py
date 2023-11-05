
import copy
import json

import numpy as np
import torch
from torch import nn
from torch_geometric.utils.unbatch import unbatch, unbatch_edge_index

from src.core.factory_base import get_instance_kvargs
from src.core.torch_base import TorchBase
from src.n_dataset.instances.graph import GraphInstance
from src.utils.cfg_utils import init_dflts_to_of, retake_oracle
from src.utils.torch.utils import rebuild_adj_matrix

#from types import NoneType



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
        
    def __infinite_data_stream(self, loader):
        # Define a generator function that yields batches of data
        while True:
            for batch in loader:
                yield batch.to(self.device)
                
    def real_fit(self):
        discriminator_loader = self.__infinite_data_stream(self.dataset.get_torch_loader(fold_id=self.fold_id, batch_size=self.batch_size, kls=self.explainee_label))
        # TODO: make it multiclass in Datase
        generator_loader = self.__infinite_data_stream(self.dataset.get_torch_loader(fold_id=self.fold_id, batch_size=self.batch_size, kls=1-self.explainee_label))

        for epoch in range(self.epochs):
            G_losses, D_losses = [], []
            #######################################################################
            self.prepare_discriminator_for_training()
            #######################################################################
            # discriminator data (real batch)
            node_features, edge_index, edge_features, _ , real_batch ,_ = next(discriminator_loader)
            # generator data (fake batch)
            fake_node_features, fake_edge_index, fake_edge_features, _ , fake_batch , _  = next(generator_loader)
            _, fake_edge_index, fake_edge_probs = self.generator(fake_node_features[1], fake_edge_index[1], fake_edge_features[1], fake_batch[1])
            # get the real and fake labels
            y_batch = torch.cat([torch.ones((len(torch.unique(real_batch[1])),)),
                                 torch.zeros(len(torch.unique(fake_batch[1])),)], dim=0).to(self.device)
            #######################################################################
            # get the oracle's predictions
            real_inst = self.__retake_batch(node_features[1], edge_index[1], edge_features[1], real_batch[1])
            fake_inst = self.__retake_batch(fake_node_features[1], fake_edge_index, fake_edge_probs, fake_batch[1], counterfactual=True, generator=True)
            oracle_scores = self.take_oracle_predictions(real_inst + fake_inst, y_batch)
            #######################################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            real_pred = self.discriminator(node_features[1], edge_index[1], edge_features[1]).expand(1)
            fake_pred = self.discriminator(fake_node_features[1], fake_edge_index, fake_edge_features[1]).expand(1)
            y_pred = torch.cat([real_pred, fake_pred])
            loss = torch.mean(self.loss_fn(y_pred.squeeze().double(), y_batch.double()) * torch.tensor(oracle_scores, dtype=torch.float))
            D_losses.append(loss.item())
            loss.backward()
            self.discriminator_optimizer.step()
            #######################################################################
            self.prepare_generator_for_training()
            ## Update G network: maximize log(D(G(z)))
            fake_features, fake_edge_index, fake_edge_attr, _, fake_batch, _ = next(generator_loader)
            y_fake = torch.ones((len(torch.unique(fake_batch[1])),)).to(self.device)
            output = self.discriminator(self.generator(fake_features[1], fake_edge_index[1], fake_edge_attr[1], fake_batch[1])[0], fake_edge_index[1], fake_edge_attr[1])
            # calculate the loss
            loss = self.loss_fn(output.expand(1).double(), y_fake.double())
            loss.backward()
            G_losses.append(loss.item())
            self.generator_optimizer.step()
                
            self.context.logger.info(f'Epoch {epoch}\t Loss_D = {np.mean(D_losses): .4f}\t Loss_G = {np.mean(G_losses): .4f}')
  
  
    def __retake_batch(self, node_features, edge_indices, edge_features, batch, counterfactual=False, generator=False):
        # unbatch edge indices
        edges = unbatch_edge_index(edge_indices, batch)
        # unbatch node_features
        node_features = unbatch(node_features, batch)
        # unbatch edge features
        if not generator:
            sizes = [index.shape[-1] for index in edges]
            edge_features = edge_features.split(sizes)
        # create the instances
        instances = []
        for i in range(len(edges)):
            if not generator:
                unbatched_edge_features = edge_features[i]
            else:
                mask = torch.zeros(edge_features.shape).to(self.device)
                mask[edges[i][0,:], edges[i][1,:]] = 1
                unbatched_edge_features = edge_features * mask
                indices = torch.nonzero(unbatched_edge_features)
                unbatched_edge_features = unbatched_edge_features[indices[:,0], indices[:,1]]
                
            instances.append(GraphInstance(id="dummy",
                                           label=1-self.explainee_label if counterfactual else self.explainee_label,
                                           data=rebuild_adj_matrix(len(node_features[i]), edges[i], unbatched_edge_features.T,self.device).detach().cpu().numpy(),
                                           node_features=node_features[i].detach().cpu().numpy(),
                                           edge_features=unbatched_edge_features.detach().cpu().numpy()))
        return instances
    
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
        
    def check_configuration(self):
        # We let TorchBase do some check for us.
        super().check_configuration()
        local_config = self.local_config
        #local_config['parameters']['epochs'] = 2000
        local_config['parameters']['batch_size'] = 1
                
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
            local_config['parameters']['disc_optimizer']['parameters']['lr']=0.001 # We override the proto lr        
        
        init_dflts_to_of(local_config, 'gen_optimizer','torch.optim.SGD',lr=0.001)
        init_dflts_to_of(local_config, 'disc_optimizer','torch.optim.SGD',lr=0.001)

        '''dataset = self.local_config['dataset']
        oracle = self.local_config['oracle']
        del self.local_config['dataset']
        del self.local_config['oracle']
        with open('GAN_dflt_'+str(self.local_config['parameters']['model_label'])+'.json', 'w') as f:
            json.dump(self.local_config,f,indent=2)
        self.local_config['dataset'] = dataset
        self.local_config['oracle'] = oracle'''