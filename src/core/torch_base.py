import numpy as np
import torch
from src.core.trainable_base import Trainable
from src.utils.cfg_utils import init_dflts_to_of
from src.core.factory_base import get_instance_kvargs

class TorchBase(Trainable):
       
    def init(self):
        self.epochs = self.local_config['parameters']['epochs']
        self.batch_size = self.local_config['parameters']['batch_size']

        
        self.model = get_instance_kvargs(self.local_config['parameters']['model']['class'],
                                   self.local_config['parameters']['model']['parameters'])

        self.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})
        
        self.loss_fn = get_instance_kvargs(self.local_config['parameters']['loss_fn']['class'],
                                           self.local_config['parameters']['loss_fn']['parameters'])
        
        
        #TODO: Need to fix GPU support!!!!
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)                            
    
    def real_fit(self):        
        loader = self.dataset.get_torch_loader(fold_id=self.fold_id, batch_size=self.batch_size, usage='train')
        
        for epoch in range(self.epochs):
            losses = []
            accuracy = []
            for batch in loader:
                batch.batch = batch.batch.to(self.device)
                node_features = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                edge_weights = batch.edge_attr.to(self.device)
                labels = batch.y.to(self.device)
                
                self.optimizer.zero_grad()
                
                pred = self.model(node_features, edge_index, edge_weights, batch.batch)
                loss = self.loss_fn(pred, labels)
                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()
                
                pred_label = torch.argmax(pred,dim=1)
                accuracy += torch.eq(labels, pred_label).long().tolist()
                
                self.optimizer.step()
            self.context.logger.info(f'epoch = {epoch} ---> loss = {np.mean(losses):.4f}\t Train accuracy = {np.mean(accuracy):.4f}')
            
    def check_configuration(self):
        super().check_configuration()
        local_config=self.local_config
        # set defaults
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 100)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 4)
        # populate the optimizer
        init_dflts_to_of(local_config, 'optimizer', 'torch.optim.Adam',lr=0.001)
        init_dflts_to_of(local_config, 'loss_fn', 'torch.nn.CrossEntropyLoss')
