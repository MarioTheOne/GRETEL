import os
import jsonpickle
#from src.utils.logger import GLogger
from logger import GLogger


class Context(object):
    __create_key = object()
    __global = None

    def __init__(self, create_key, conf_file):
        assert(create_key == Context.__create_key), \
            "Context objects must be created using Context.get_context"
              
        # Check that the path to the config file exists
        if not os.path.exists(conf_file):
            raise ValueError(f'''The provided config file does not exist. PATH: {conf_file}''')

        # Read the config dictionary inside the config path
        with open(conf_file, 'r') as config_reader:
            self.conf = jsonpickle.decode(config_reader.read())

        self._scope = self.conf['experiment']['scope']

        GLogger._path=os.path.join(self._get_store_path("output_store_path"),self._scope,"logs")
        self.logger = GLogger.getLogger()
        self.logger.info("Successfully created the context of '%s'", self._scope) 

        
    @classmethod
    def get_context(cls,conf_file=None):
        if(Context.__global == None):
            if conf_file is None:
                raise ValueError(f'''The configuration file must be passed to the method as PATH the first time. Now you did not pass as parameter.''')
            Context.__global = Context(Context.__create_key,conf_file)            
        return Context.__global
    
    def _get_store_path(self,value):
        return self._get_by_pkvs("store_paths","name",value,"address")

    def _get_by_pkvs(self,parent,key,value,son):
        for obj in self.conf[parent]:
            if(obj[key] == value):
                return obj[son]

#context = Context.get_context()
context = Context.get_context("config/test/dummy_conf.json")
print(context._scope)


            