class DataInstance:
    
    def __init__(self, id, label, data, dataset=None):
        self.id = id
        self.data = data
        self.label = label
        # empty dataset by default
        self._dataset = dataset
    
    @property
    def dataset(self):
        return self._dataset
    
    @dataset.setter
    def dataset(self, new_dataset):
        self._dataset = new_dataset
        
    
    