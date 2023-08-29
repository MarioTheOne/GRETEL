import os

import jsonpickle
import numpy as np
import torch
from torch_geometric.data import DataLoader

from src.dataset.dataset_base import Dataset
from src.dataset.torch_geometric.dataset_geometric import TorchGeometricDataset
from src.oracle.oracle_base import Oracle


class OracleTorch(Oracle):
    
    def __init__(self, context, local_config) -> None:
        super().__init__(context)
        
        self.name = self.__class__.__name__
        
        self.logger = context.logger
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.converter = converter
        self.batch_size = batch_size
        
        self.model = network
        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        
    
    def embedd(self, instance):
        return instance                                 
            
    def real_fit(self, dataset: Dataset, fold_id=0):
        # If there is an available oracle trained on that dataset load it
        if os.path.exists(os.path.join(self._oracle_store_path, self._name)):
            self.read_oracle()
        else:
            dataset = self.converter.convert(dataset)
            loader = self.transform_data(dataset, fold_id=fold_id, usage='train')
            
            for epoch in range(self.epochs):
                
                losses = []
                for batch in loader:
                    node_features = batch.x.to(self.device)
                    edge_index = batch.edge_index.to(self.device)
                    edge_weights = batch.edge_attr.to(self.device)
                    labels = batch.y.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    pred = self.network(node_features, edge_index, edge_weights)
                    
                    loss = self.loss_fn(pred, labels)
                    losses.append(loss.to('cpu').detach().numpy())
                    loss.backward()
                    
                    self.optimizer.step()
                
                self.logger.info(f'epoch = {epoch} ---> loss = {np.mean(losses):.4d}')
            # Writing to disk the trained oracle
            self.write_oracle()
            #self.evaluate(dataset, fold_id=fold_id)
            
    @torch.no_grad()        
    def evaluate(self, dataset: Dataset, fold_id=0):
        # If there is an available oracle trained on that dataset load it
        if os.path.exists(os.path.join(self._oracle_store_path, self._name)):
            self.read_oracle()
        else:
            raise FileExistsError(f'The Oracle {self._name} must be trained first.')
            
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
            
            pred = self.network(node_features, edge_index, edge_weights)
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
            "model" : self.network.state_dict(),
            "config": self.local_config
        }
        
        with open(os.path.join(directory, 'oracle'), 'w') as f:
            f.write(jsonpickle.encode(dump))
      
    def read(self):
        dump_file = os.path.join(self.context.oracle_store_path, self.name, 'oracle')
        
        if os.path.exists(dump_file):
            with open(dump_file, 'r') as f:
                dump = jsonpickle.decode(f.read())
                self.network.load_state_dict(dump['model'])
                self.local_config = dump['config']