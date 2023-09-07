from src.core.factory_base import Factory
from src.core.explainer_base import Explainer
from typing import List

class ExplainerFactory(Factory):
          
    def get_explainer(self, explainer_snippet):
        return self._get_object(explainer_snippet)
            
    def get_explainers(self, config_list) -> List[Explainer]:
        return [self.get_explainer(obj) for obj in config_list]