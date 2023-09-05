from src.core.oracle_base import Oracle

# A CustomOracle assume that it is a kind of "static" function.
# Thus only the functions:
#   def _real_predict(self, data_instance):
#   def _real_predict_proba(self, data_instance):
# must be implemented. 
# The remaing functions are just empty implementation because are not needed.
class CustomOracle(Oracle):

    def check_configuration(self, local_config):
        return local_config