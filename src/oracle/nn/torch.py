import os

import pickle
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.dataset.dataset_base import Dataset
from src.dataset.torch_geometric.dataset_geometric import TorchGeometricDataset
from src.core.oracle_base import Oracle
from src.utils.utils import get_instance_kvargs, config_default

class OracleTorch(Oracle):
       
    def init(self):
        self.epochs = self.local_config['parameters']['epochs']
        
        self.model = get_instance_kvargs(self.local_config['parameters']['model']['class'],
                                   self.local_config['parameters']['model']['parameters'])

        self.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})
        
        self.loss_fn = get_instance_kvargs(self.local_config['parameters']['loss_fn']['class'],
                                        self.local_config['parameters']['loss_fn']['parameters'])
        
        self.converter = get_instance_kvargs(self.local_config['parameters']['converter']['class'],
                                      self.local_config['parameters']['converter']['parameters'])
        
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
            
    def real_fit(self):
        fold_id = self.local_config['parameters']['fold_id']
        loader = self._transform_data(self.dataset, fold_id=fold_id, usage='train')
        
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
                accuracy += torch.eq(labels, pred_label).int().tolist()
                
                self.optimizer.step()
            self.context.logger.info(f'epoch = {epoch} ---> loss = {np.mean(losses):.4f}\t Train accuracy = {np.mean(accuracy):.4f}')
        self.evaluate(self.dataset, fold_id=fold_id)
            
    @torch.no_grad()        
    def evaluate(self, dataset: Dataset, fold_id=0):            
        #dataset = self.converter.convert(dataset)
        loader = self._transform_data(dataset, fold_id=fold_id, usage='test')
        
        losses = []
        accuracy = []
        for batch in loader:
            node_features = batch.x.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            edge_weights = batch.edge_attr.to(self.device)
            labels = batch.y.to(self.device)
            
            self.optimizer.zero_grad()  
            pred = self.model(node_features, edge_index, edge_weights, batch.batch)
            
            loss = self.loss_fn(pred, labels)
            losses.append(loss.to('cpu').detach().numpy())
            
            pred_label = torch.argmax(pred,dim=1)
            accuracy += torch.eq(labels, pred_label).int().tolist()
        
        self.context.logger.info(f'Test accuracy ---> Test accuracy = {np.mean(accuracy):.4f}')


    def _real_predict(self, data_instance):
        return  torch.argmax(self._real_predict_proba(data_instance))
    
    @torch.no_grad()
    def _real_predict_proba(self, data_instance):       
        data_inst = TorchGeometricDataset([data_instance])

        data = data_inst.instances[0]
        node_features = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_weights = data.edge_attr.to(self.device)

        return self.model(node_features,edge_index,edge_weights, None).squeeze()
        
        
    def _transform_data(self, dataset: Dataset, fold_id=-1, usage='train'):                     
        indices = dataset.get_split_indices(fold_id)[usage]
        data_list = [inst for inst in dataset.instances if inst.id in indices]
        dgl_dataset = TorchGeometricDataset(data_list)        
        dataloader = DataLoader(dgl_dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device=self.device))
        return dataloader
    
                     
    def check_configuration(self, local_config):
        local_config['parameters'] = local_config.get('parameters', {})
        
        # set defaults
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 100)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 8)
        
        # populate the optimizer
        config_default(local_config, 'optimizer', 'torch.optim.Adam')
        config_default(local_config, 'loss_fn', 'torch.nn.CrossEntropyLoss')
        config_default(local_config, 'converter', 'src.dataset.converters.weights_converter.DefaultFeatureAndWeightConverter')
        
        return local_config
