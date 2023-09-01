from src.utils.utils import get_instance, add_init_defaults_params
import json

def check_configuration(local_config):
        local_config['parameters'] = local_config.get('parameters', {})
        
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 100)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 8)
        print(local_config)
        # populate the optimizer
        __config_helper(local_config, 'optimizer', 'torch.optim.Adam')
        __config_helper(local_config, 'loss_fn', 'torch.nn.BCELoss')
        __config_helper(local_config, 'converter', 'src.dataset.converters.weights_converter.DefaultFeatureAndWeightConverter')
        
        return local_config
    
    
def __config_helper(node, key, kls):
    if key not in node['parameters']:
        node['parameters'][key] = {
            "class": kls, 
            "parameters": { }
        }
        node_config = add_init_defaults_params(kls, node['parameters'][key]['parameters'])
        node['parameters'][key]['parameters'] = node_config



if __name__ == '__main__':

    icfg = {
  "parameters": {
    "epochs": 100,
    "batch_size": 8,
    "optimizer": {
      "class": "torch.optim.Adam",
      "parameters": {
        "lr": 0.001,
        "betas": [
          0.09,
          0.999
        ],
        "eps": 1e-08,
        "weight_decay": 0,
        "amsgrad": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False
      }
    },
    "loss_fn": {
      "class": "torch.nn.BCELoss",
      "parameters": {     
        "reduction": "mean"
      }
    },
    "converter": {
      "class": "src.dataset.converters.weights_converter.DefaultFeatureAndWeightConverter",
      "parameters": {
        "feature_dim": 8,
        "weight_dim": 1
      }
    }
  }
}
    print(json.dumps(check_configuration(icfg), indent=4))