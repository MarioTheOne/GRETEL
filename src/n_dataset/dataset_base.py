import pickle
from typing import List
from src.core.savable import Savable
from src.n_dataset.instances.base import DataInstance
from src.utils.context import Context
from src.utils.utils import get_instance_kvargs, build_default_config_obj
from sklearn.model_selection import StratifiedKFold

class Dataset(Savable):
    
    def __init__(self, context:Context, local_config) -> None:
        super().__init__(context, local_config)
        self.instances: List[DataInstance] = []
        
        self.node_features_map = {}
        self.edge_features_map = {}
        self.graph_features_map = {}
        
        self.splits = []
        
        self.check_configuration(self.local_config)
        self.load_or_save()
        
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
            
        self.generate_splits(n_splits=self.local_config['parameters']['n_splits'],
                             shuffle=self.local_config['parameters']['shuffle'])
    
        self.write()
        
    def get_data(self):
        return self.instances
    
    def get_instance(self, i: int):
        return self.instances[i]
    
    def get_split_indices(self, fold_id=-1):
        print(self.splits)
        if fold_id == -1:
            return {'train': list(self.splits[0].values()), 'test': []}
        else:
            return self.splits[fold_id]
    
    def generate_splits(self, n_splits=10, shuffle=True):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
        spl = kf.split([g for g in self.instances], [g.label for g in self.instances])
        for train_index, test_index in spl:
            self.splits.append({'train': train_index, 'test': test_index})
    
    def read(self):
        if self.saved():
            store_path = self.context.get_path(self)
            with open(store_path, 'rb') as f:
                dump = pickle.load(f)
                self = dump['dataset']
                self.local_config = dump['config']
                print(self.splits[0])
                
    def write(self):
        store_path = self.context.get_path(self)
        
        dump = {
            "dataset" : self,
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