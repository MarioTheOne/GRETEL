import os

import pickle
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.dataset.dataset_base import Dataset
from src.dataset.torch_geometric.dataset_geometric import TorchGeometricDataset
from src.core.oracle_base import Oracle
from src.utils.utils import get_instance_kvargs, add_init_defaults_params

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
        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )                              
            
    def real_fit(self):
        fold_id = self.local_config['parameters']['fold_id']
        dataset = self.converter.convert(self.dataset)
        loader = self._transform_data(dataset, fold_id=fold_id, usage='train')
        
        for epoch in range(self.epochs):
            
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
                loss.backward()
                
                pred_label = torch.argmax(pred,dim=1)
                accuracy += torch.eq(labels, pred_label).int().tolist()
                
                self.optimizer.step()
            self.context.logger.info(f'epoch = {epoch} ---> loss = {np.mean(losses):.4f}\t Train accuracy = {np.mean(accuracy):.4f}')
        self.evaluate(dataset, fold_id=fold_id)
            
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
        data_instance = self.converter.convert_instance(data_instance)
        data_inst = TorchGeometricDataset([data_instance])

        data = data_inst.instances[0]
        node_features = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_weights = data.edge_attr.to(self.device)

        return self.model(node_features,edge_index,edge_weights, None).squeeze()
        
        
    def _transform_data(self, dataset: Dataset, fold_id=-1, usage='train'):                     
        indices = dataset.get_split_indices()[fold_id][usage]
        data_list = [inst for inst in dataset.instances if inst.id in indices]
        dgl_dataset = TorchGeometricDataset(data_list)
        dataloader = DataLoader(dgl_dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader
    
    
    def write(self):
        filepath = self.context.get_path(self)
            
        dump = {
            "model" : self.model.state_dict(),
            "config": self.local_config
        }
        
        with open(filepath, 'wb') as f:
          pickle.dump(dump, f)
      
    def read(self):
        dump_file = self.context.get_path(self)
        
        if os.path.exists(dump_file):
            with open(dump_file, 'rb') as f:
                dump = pickle.load(f)
                self.model.load_state_dict(dump['model'])
                self.local_config = dump['config']

                     
    def check_configuration(self, local_config):
        local_config['parameters'] = local_config.get('parameters', {})
        
        # set defaults
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 100)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 8)
        
        # populate the optimizer
        self.__config_helper(local_config, 'optimizer', 'torch.optim.Adam')
        self.__config_helper(local_config, 'loss_fn', 'torch.nn.CrossEntropyLoss')
        self.__config_helper(local_config, 'converter', 'src.dataset.converters.weights_converter.DefaultFeatureAndWeightConverter')
        
        return local_config
    
    #TODO: Generalize this function and made it available to all.
    def __config_helper(self, node, key, kls):
        if key not in node['parameters']:
            node['parameters'][key] = {
                "class": kls, 
                "parameters": { }
            }

        node_config = add_init_defaults_params(kls, node['parameters'][key]['parameters'])
        node['parameters'][key]['parameters'] = node_config
