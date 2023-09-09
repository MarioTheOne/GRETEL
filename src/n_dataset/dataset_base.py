import pickle
from typing import List

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from src.core.factory_base import get_instance_kvargs

from src.core.savable import Savable
from src.n_dataset.instances.base import DataInstance
from src.n_dataset.utils.dataset_torch import TorchGeometricDataset
from src.utils.context import Context
from src.core.factory_base import get_class


class Dataset(Savable):
    
    def __init__(self, context:Context, local_config) -> None:
        super().__init__(context, local_config)
        
        ################### PREAMBLE ###################
        self.instances: List[DataInstance] = []
        
        self.node_features_map = {}
        self.edge_features_map = {}
        self.graph_features_map = {}
        
        self.splits = []
        self._torch_repr = None
        self._class_indices = {}
        self._num_nodes = None
        #################################################
        
        #TODO: I guess we can  move before everything and including it in the super class ->
        self.check_configuration(self.local_config)
        self.load_or_create()
        
    def create(self):
        self.generator = get_instance_kvargs(self.local_config['parameters']['generator']['class'],
                                                {
                                                    "context": self.context, 
                                                    "local_config": self.local_config['parameters']['generator'],
                                                    "dataset": self
                                                })
        
        for manipulator in self.local_config['parameters']['manipulators']:
            get_instance_kvargs(manipulator['class'],
                                {
                                    "context": self.context,
                                    "local_config": manipulator,
                                    "dataset": self
                                })
            self.context.logger.info("Instantiated: "+manipulator['class'])
            
        self.generate_splits(n_splits=self.local_config['parameters']['n_splits'],
                             shuffle=self.local_config['parameters']['shuffle'])
    
        self.write()

        
    def get_data(self):
        return self.instances
    
    def get_instance(self, i: int):
        return self.instances[i]
    
    def num_node_features(self):
        return len(self.node_features_map)
    
    def num_edge_features(self):
        return len(self.edge_features_map)
    
    def num_graph_features(self):
        return len(self.graph_features_map)
    
    def class_indices(self):
        if not self._class_indices:
            for i, inst in enumerate(self.instances):
                self._class_indices[inst.label] = self._class_indices.get(inst.label, []) + [i]
        return self._class_indices
    
    @property        
    def num_classes(self):
        return len(self.class_indices())
    
    @property
    def num_nodes(self):
        if not self._num_nodes:
            self._num_nodes = len(self.get_instance(0).data)
        return self._num_nodes
    
    def get_split_indices(self, fold_id=-1):
        if fold_id == -1:
            #NOTE: i am bit worried that it might be weak if you have sparse indices
            return {'train': list(range(0, len(self.instances))), 'test': list(range(0, len(self.instances))) }
        else:
            return self.splits[fold_id]
    
    def generate_splits(self, n_splits=10, shuffle=True):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
        spl = kf.split([g for g in self.instances], [g.label for g in self.instances])
        for train_index, test_index in spl:
            self.splits.append({'train': train_index.tolist(), 'test': test_index.tolist()})
            
    def get_torch_loader(self, fold_id=-1, batch_size=4, usage='train', kls=-1):
        if not self._torch_repr:
            self._torch_repr = TorchGeometricDataset(self.instances)
        # get the train/test indices from the dataset
        indices = self.get_split_indices(fold_id)[usage]
        # get only the indices of a specific class
        if kls != -1:
            indices = list(set(indices).difference(set(self.class_indices()[kls])))
        return DataLoader(Subset(self._torch_repr, indices), batch_size=batch_size, shuffle=True)
    
    def read(self):
        if self.saved():
            store_path = self.context.get_path(self)
            with open(store_path, 'rb') as f:
                dump = pickle.load(f)
                self.instances = dump['instances']
                self.splits = dump['splits']
                self.local_config = dump['config']
                
    def write(self):
        store_path = self.context.get_path(self)
        
        dump = {
            "instances" : self.instances,
            "splits": self.splits,
            "config": self.local_config
        }
        
        with open(store_path, 'wb') as f:
            pickle.dump(dump, f)
            
            
    def check_configuration(self, local_config):
        if 'generator' not in local_config['parameters']:
            raise ValueError(f'''The "generator" parameter needs to be specified in {self}''')
        
        if 'manipulators' not in local_config['parameters']: # or not len(local_config['parameters']['manipulators']):
            local_config['parameters']['manipulators'] = []
        
        #local_config['parameters']['manipulators'].append(build_default_config_obj("src.n_dataset.manipulators.centralities.NodeCentrality"))
        #local_config['parameters']['manipulators'].append(build_default_config_obj("src.n_dataset.manipulators.weights.EdgeWeights"))
            
        local_config['parameters']['n_splits'] = local_config['parameters'].get('n_splits', 10)
        local_config['parameters']['shuffle'] = local_config['parameters'].get('shuffle', True)
        
        return local_config
    
    @property
    def name(self):
        alias = get_class( self.local_config['parameters']['generator']['class'] ).__name__
        return self.context.get_name(self,alias=alias)