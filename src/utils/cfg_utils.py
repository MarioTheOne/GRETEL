

import inspect
from src.core.factory_base import get_class

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
    default_embedder_cls = get_class(snippet['class'])
    # get the parameters of the constructor of the desired class
    # and skip the self class parameter that isn't useful
    sg = inspect.signature(default_embedder_cls.__init__)
    default_params = [(p.name, p.default) for p in sg.parameters.values() if ((p.default != p.empty) and p.default != None)]
    embedder_cls_params = dict(default_params)
    embedder_params = snippet['parameters']
    # update the embedder params with only those
    # values that haven't been specified and have a default value
    snippet['parameters'] = {**embedder_cls_params, **kwargs, **embedder_params}

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
