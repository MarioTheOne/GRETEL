from abc import ABCMeta
from src.core.grtl_base import Base
from src.utils.utils import get_class


class Factory(Base,metaclass=ABCMeta):      
    def _get_object(self, object_snippet):
        base_obj = get_class(object_snippet['class'])(self.context, object_snippet)
        self.context.logger.info("Created: "+ str(base_obj))
        return base_obj
