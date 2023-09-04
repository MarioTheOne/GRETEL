import pickle
from typing import List

from src.core.savable import Savable
from src.n_dataset.instances.base import DataInstance
from src.utils.context import Context
from src.utils.utils import get_instance


class Dataset(Savable):
    
    def __init__(self, context:Context, local_config) -> None:
        super().__init__(context, local_config)
        self._instance_id_counter = 0
        self.instances: List[DataInstance] = []
        
        if 'loader' in self.local_config['parameters']:
            self.loader = get_instance(self.local_config['parameters']['loader']['class'],
                                       context, local_config['parameters']['loader'], self)
        
    def get_data(self):
        return self.instances
    
    def get_instance(self, i: int):
        return self.instances[i]
    
    def read(self):
        if self.saved():
            store_path = self.context.get_path(self)
            with open(store_path, 'rb') as f:
                loaded_dataset = pickle.load(f)
                self.instances = loaded_dataset.instances
                self._instance_id_counter = len(self.instances)
    
    def write(self):
        store_path = self.context.get_path(self)
        with open(store_path, 'wb') as f:
            pickle.dump(self, f)