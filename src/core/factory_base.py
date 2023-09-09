from abc import ABCMeta
from src.core.grtl_base import Base

class Factory(Base,metaclass=ABCMeta):    
      
    def _get_object(self, object_snippet):
        base_obj = get_class(object_snippet['class'])(self.context, object_snippet)
        self.context.logger.info("Created: "+ str(base_obj))
        return base_obj
    
    def check_configuration(self, local_config):
        return local_config
    
    def get_default_cfg(self):
        pass


################ Utilities functions for Object creation ################


def get_instance_kvargs(kls, param):
    return get_class(kls)(**param)


def get_instance(kls, param):
    return get_class(kls)(param)

def get_class( kls ):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)            
    return m
