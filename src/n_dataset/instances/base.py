class DataInstance:
    
    def __init__(self, id, label, data):
        self.id = id
        self.data = data
        self.label = label #TODO: Refactoring to have a one-hot encoding of labels!
        
    @property
    def num_nodes(self):
        return len(self.data)
        
    
    