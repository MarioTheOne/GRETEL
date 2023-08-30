from src.dataset.dataset_base import Dataset
from src.core.trainable_base import Trainable

class Oracle(Trainable):

    def __init__(self, context, local_config) -> None:
        super().__init__(context, local_config)
        self._call_counter = 0

    def retrain(self):
        self.fit()
        
    def init_model(self):
        raise NotImplementedError()

    def real_fit(self):
        raise NotImplementedError()
    
    def evaluate(self, dataset: Dataset, fold_id=0):
        raise NotImplementedError()
    
    def predict(self, data_instance):
        """predicts the label of a given data instance
        -------------
        INPUT:
            data_instance : The instance whose class is going to be predicted 
        -------------
        OUTPUT:
            The predicted label for the data instance
        """
        self._call_counter += 1

        return self._real_predict(self.embedd(data_instance))

    
    def predict_proba(self, data_instance):
        """predicts the probability estimates for a given data instance
        -------------
        INPUT:
            data_instance : The instance whose class is going to be predicted 
        -------------
        OUTPUT:
            The predicted probability estimates for the data instance
        """
        self._call_counter += 1

        return self._real_predict_proba(self.embedd(data_instance))
    
    def predict_list(self, dataset: Dataset, fold_id=0):

        sptest = dataset.get_split_indices()[fold_id]['test']

        result = [self.predict(dataset.get_instance(i)) for i in sptest]

        return result

    def check_configuration(self, local_config):
        return NotImplementedError()
    
    def _real_predict(self, data_instance):
        raise NotImplementedError()
    
    def _real_predict_proba(self, data_instance):
        raise NotImplementedError()
    
    def embedd(self, instance):
        raise NotImplementedError()

    def get_calls_count(self):
        return self._call_counter

    def reset_call_count(self):
        self._call_counter = 0
