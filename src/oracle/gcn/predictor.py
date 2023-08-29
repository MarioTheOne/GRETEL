
from src.oracle.torch.predictor import OracleTorch


class GCNOracle(OracleTorch):
    
    def __init__(self, id, oracle_store_path, network,
                 converter, optimizer, loss_fn, epochs=100,
                 batch_size=8, config_dict=None) -> None:
        
        super().__init__(id, oracle_store_path, network,
                         converter, optimizer, loss_fn,
                         epochs=epochs, batch_size=batch_size,
                         config_dict=config_dict)
        
        self._name = 'cf2'
        
