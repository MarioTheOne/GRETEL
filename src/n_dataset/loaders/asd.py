from os import listdir
from os.path import isfile, join

import numpy as np

from src.n_dataset.instances.graph import GraphInstance
from src.n_dataset.loaders.base import Loader


class ASDLoader(Loader):
    
    def get_num_instances(self):
        return len(self.dataset.instances)
    
    def init(self):
        base_path = self.local_config['parameters']['path']
        td_file_path = join(base_path, 'td')
        asd_file_path = join(base_path, 'asd')
        
        for label, dir in enumerate([td_file_path, asd_file_path]):
            self.read(dir, label=label)
        
    def read(self, path, label=0):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        instance_id = 0
        for file in files:
            instance = [[int(num) for num in line.split(' ')] for line in file]
            # Creating the instance
            inst = GraphInstance(instance_id)
            instance_id += 1    
            inst.label = label
            # transforming the numpy array into a graph
            inst.data = np.array(instance, dtype=np.int32)
            # Adding the instance to the instances list
            self.dataset.instances.append(inst)
            instance_id += 1
        
