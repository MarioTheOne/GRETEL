import inspect
import json
from src.core.factory_base import get_class

def pprint(dic):
    print(json.dumps(dic, indent=4))


def inject_dataset(cfg, dataset):
    cfg['parameters'] = cfg.get('parameters',{})
    cfg['parameters']['dataset']= dataset

def inject_oracle(cfg, oracle):
    cfg['parameters'] = cfg.get('parameters',{})
    cfg['parameters']['oracle']= oracle

def retake_dataset(cfg):
    cfg['parameters'] = cfg.get('parameters',{})
    return cfg['parameters']['dataset']

def retake_oracle(cfg):
    cfg['parameters'] = cfg.get('parameters',{})
    return cfg['parameters']['oracle']

def add_init_defaults_params(snippet, **kwargs):
    declared_cls = get_class(snippet['class'])
    user_defined_params = snippet['parameters']
    # get the parameters of the constructor of the desired class
    # and skip the self class parameter that isn't useful
    sg = inspect.signature(declared_cls.__init__)
    signature_params = [(p.name, p.default) for p in sg.parameters.values() if ((p.default != p.empty) and p.default != None)]
    default_cls_params = dict(signature_params)
    
    # update the user defined params with only those
    # values that haven't been specified and have a default value
    # in any case override the user parameters with the code passed one
    snippet['parameters'] = {**default_cls_params, **user_defined_params, **kwargs}

def init_dflts_to_of(snippet, key, kls, *args, **kwargs):
    __add_dflts_to_of(snippet, key, kls, generate_default_for,*args, **kwargs)

def get_dflts_to_of(snippet, key, kls, *args, **kwargs):
    __add_dflts_to_of(snippet, key, kls, empty_cfg_for, *args, **kwargs)

def __add_dflts_to_of(snippet, key, kls, func, *args, **kwargs):
    if key not in snippet['parameters']:
        snippet['parameters'][key] = __get_default_for(kls, func,*args, **kwargs)
    else:
        add_init_defaults_params(snippet['parameters'][key],**kwargs)

def __get_default_for(kls,func,*args, **kwargs):
    methods = [method[1] for method in inspect.getmembers(get_class(kls)) \
               if hasattr(method[1], '__name__') and method[1].__name__ == default_cfg.__name__]
    return methods[0](kls, *args, **kwargs) if len(methods)>0 else func(kls,**kwargs)

def generate_default_for(kls, **kwargs):
    cfg = empty_cfg_for(kls, **kwargs)
    add_init_defaults_params(cfg, **kwargs)
    return cfg

def empty_cfg_for(kls,**kwargs):
    return {"class": kls, "parameters": { } }

def set_if_not(snippet, key, subsnip):
    if key not in snippet['parameters']:
        snippet['parameters'][key] = subsnip

# The following is an annotation that made the method @static
def default_cfg(func):
    @staticmethod
    def default_cfg(*args, **kwargs):        
        return func(*args, **kwargs)
    return default_cfg
