from src.core.grtl_base import Base

from abc import ABCMeta, abstractmethod

from src.utils.context import Context


class Explainer(Base, metaclass=ABCMeta):
    
    def __init__(self, context: Context, local_config=None) -> None:
        super().__init__(context, local_config)
        self.oracle = self.local_config['oracle']
        self.dataset = self.local_config['dataset']

    @abstractmethod
    def explain(self, instance):
        pass
