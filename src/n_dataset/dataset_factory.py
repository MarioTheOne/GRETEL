from src.core.factory_base import Factory
from src.n_dataset.dataset_base import Dataset
from typing import List

class DatasetFactory(Factory):
    
    def get_dataset(self, dataset_snippet):
        return self._get_object(dataset_snippet)
            
    def get_datasets(self, config_list) -> List[Dataset]:
        return [self.get_dataset(obj) for obj in config_list]