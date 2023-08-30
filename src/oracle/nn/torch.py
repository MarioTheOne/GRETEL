import os

import jsonpickle
import numpy as np
import torch
from torch_geometric.data import DataLoader

from src.dataset.dataset_base import Dataset
from src.dataset.torch_geometric.dataset_geometric import TorchGeometricDataset
from src.oracle.oracle_base import Oracle
from src.utils.utils import get_instance, get_instance_kvargs, get_only_default_params

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
        
        
                
    def embedd(self, instance):
        return instance                                 
            
    def real_fit(self):
        fold_id = self.local_config['parameters']['fold_id']
        dataset = self.converter.convert(self.dataset)
        loader = self.transform_data(dataset, fold_id=fold_id, usage='train')
        
        for epoch in range(self.epochs):
            
            losses = []
            for batch in loader:
                node_features = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                edge_weights = batch.edge_attr.to(self.device)
                labels = batch.y.to(self.device)
                
                self.optimizer.zero_grad()
                
                pred = self.model(node_features, edge_index, edge_weights)
                
                loss = self.loss_fn(pred, labels)
                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()
                
                self.optimizer.step()
            
            self.context.logger.info(f'epoch = {epoch} ---> loss = {np.mean(losses):.4d}')
        #self.evaluate(dataset, fold_id=fold_id)
            
    @torch.no_grad()        
    def evaluate(self, dataset: Dataset, fold_id=0):            
        dataset = self.converter.convert(dataset)
        loader = self.transform_data(dataset, fold_id=fold_id, usage='test')
        
        losses = []
        accuracy = []
        for batch in loader:
            node_features = batch.x.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            edge_weights = batch.edge_attr.to(self.device)
            labels = batch.y.to(self.device)
            # n x 1   
            self.optimizer.zero_grad()  
            pred = self.model(node_features, edge_index, edge_weights)
            # n x k
            loss = self.loss_fn(pred, labels)
            losses.append(loss.to('cpu').detach().numpy())
        
        
    def transform_data(self, dataset: Dataset, fold_id=-1, usage='train'):             
        adj  = np.array([i.to_numpy_array() for i in dataset.instances])
        features = np.array([i.features for i in dataset.instances])
        weights = np.array([i.weights for i in dataset.instances])
        y = np.array([i.graph_label for i in dataset.instances])
        
        indices = dataset.get_split_indices()[fold_id][usage]
        
        adj = adj[indices]
        features = features[indices]
        weights = weights[indices]
        y = y[indices] 
              
        dgl_dataset = TorchGeometricDataset(adj, features, weights, y)
        dataloader = DataLoader(dgl_dataset, batch_size=self.batch_size, shuffle=True)

        return dataloader
    
    
    def write(self):
        directory = os.path.join(self.context.oracle_store_path, self.name)
        if not os.path.exists(directory):
            os.mkdir(directory)
            
        dump = {
            "model" : self.model.state_dict(),
            "config": self.local_config
        }
        
        with open(os.path.join(directory, 'oracle'), 'w') as f:
            f.write(jsonpickle.encode(dump))
      
    def read(self):
        dump_file = os.path.join(self.context.oracle_store_path, self.name, 'oracle')
        
        if os.path.exists(dump_file):
            with open(dump_file, 'r') as f:
                dump = jsonpickle.decode(f.read())
                self.model.load_state_dict(dump['model'])
                self.local_config = dump['config']
                
                
    def check_configuration(self, local_config):
        local_config['parameters'] = local_config.get('parameters', {})
        
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 100)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 8)
        
        # populate the optimizer
        #TODO: local_config['parameters'] --> local_config in the following 
        self.__config_helper(local_config, 'optimizer', 'torch.optim.Adam')
        self.__config_helper(local_config, 'loss_fn', 'torch.nn.BCELoss')
        self.__config_helper(local_config, 'converter', 'src.dataset.converters.weights_converter.DefaultFeatureAndWeightConverter')
        
        return local_config
    
    
    def __config_helper(self, node, key, kls):
        if key not in node['parameters']:
            node['parameters'][key] = {
                "class": kls, 
                "parameters": { }
            }
        #TODO: revise get_only_default_params: Actually it reurn Null parameters and false (wit lowercase F). It might be a problem. 
        node_config = get_only_default_params(kls, node['parameters'][key]['parameters'])
        node['parameters'][key]['parameters'] = node_config
