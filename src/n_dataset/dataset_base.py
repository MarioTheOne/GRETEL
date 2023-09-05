import pickle
from typing import List

from src.core.savable import Savable
from src.n_dataset.instances.base import DataInstance
from src.utils.context import Context
from src.utils.utils import get_instance_kvargs


class Dataset(Savable):
    
    def __init__(self, context:Context, local_config) -> None:
        super().__init__(context, local_config)
        self._instance_id_counter = 0
        self.instances: List[DataInstance] = []
        
        self.node_features_map = {}
        self.edge_features_map = {}
        self.graph_features_map = {}
        
        if 'generator' in self.local_config['parameters']:
            self.generator = get_instance_kvargs(self.local_config['parameters']['generator']['class'],
                                                 {
                                                     "context": self.context, 
                                                     "local_config": self.local_config['parameters']['generator'],
                                                     "dataset": self
                                                 })
        self.write()
        
    def get_data(self):
        return self.instances
    
    def get_instance(self, i: int):
        return self.instances[i]
    
    def read(self):
        if self.saved():
            store_path = self.context.get_path(self)
            with open(store_path, 'rb') as f:
                self.instances = pickle.load(f)
                self._instance_id_counter = len(self.instances)
    
    def write(self):
        store_path = self.context.get_path(self)
        with open(store_path, 'wb') as f:
            pickle.dump(self.instances, f)
