from os import listdir
from os.path import isfile, join

import numpy as np

from src.n_dataset.instances.graph import GraphInstance
from src.n_dataset.generators.base import Generator


class ASDGenerator(Generator):
    
    def get_num_instances(self):
        return len(self.dataset.instances)
    
    def generate_dataset(self):
        if not len(self.dataset.instances):
            for label, dir in enumerate([self._td_file_path, self._asd_file_path]):
                self.read(dir, label=label)
    
    def init(self):
        base_path = self.local_config['parameters']['data_dir']
        self._td_file_path = join(base_path, 'td')
        self._asd_file_path = join(base_path, 'asd')
        self.generate_dataset()
        
    def read(self, path, label=0):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        instance_id = 0
        for filename in files:
            with open(join(path, filename), 'r') as f:
                instance = [[int(num) for num in line.split(' ')] for line in f]
                self.dataset.instances.append(GraphInstance(instance_id, label=label, data=np.array(instance, dtype=np.int32)))
                instance_id += 1    
        
