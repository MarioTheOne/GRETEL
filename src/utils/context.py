import inspect
import os

import jsonpickle
from src.utils.composer import compose
from src.utils.logger import GLogger


class Context(object):
    __create_key = object()
    __global = None

    def __init__(self, create_key, config_file):
        ###################################################
        self.factories = {
            "datasets": None,
            "embedders": None,
            "oracles": None,
            "explainers": None,
            "metrics": None
        }
        ###################################################
        assert(create_key == Context.__create_key), \
            "Context objects must be created using Context.get_context"
              
        # Check that the path to the config file exists
        if not os.path.exists(config_file):
            raise ValueError(f'''The provided config file does not exist. PATH: {config_file}''')

        self.config_file = config_file
        # Read the config dictionary inside the config path with the composer
        with open(self.config_file, 'r') as config_reader:
            self.conf = compose(jsonpickle.decode(config_reader.read()))

        self._scope = self.conf['experiment']['scope']

        GLogger._path = os.path.join(self.log_store_path, self._scope)
        self.logger = GLogger.getLogger()
        self.logger.info("Successfully created the context of '%s'", self._scope)
        
        self.__create_storages()
        
    @classmethod
    def get_context(cls,config_file=None):
        if(Context.__global == None):
            if config_file is None:
                raise ValueError(f'''The configuration file must be passed to the method as PATH the first time. Now you did not pass as parameter.''')
            Context.__global = Context(Context.__create_key,config_file)            
        return Context.__global
    
    @classmethod
    def get_by_pkvs(cls, conf, parent, key, value, son):
        for obj in conf[parent]:
            if(obj[key] == value):
                return obj[son]
    
    def get_path(self, obj):
        fullname = self.get_fullname(obj).split('.')
        qualifier = fullname[1] + '_store_path'
        #dataset_path = self.get_name(obj.dataset.__class__.__name__, obj.dataset.local_config)
        # change this path when the dataset factories are finished
        directory = os.path.join(self._get_store_path(qualifier), obj.dataset.__class__.__name__)
        if not os.path.exists(directory):
            os.mkdir(directory)
        return os.path.join(directory, obj.name)
    
    def get_fullname(self, o):
        klass = o.__class__
        module = klass.__module__
        if module == 'builtins':
            return klass.__qualname__ # avoid outputs like 'builtins.str'
        return module + '.' + klass.__qualname__
        
    def get_name(self, cls, dictionary):
        
        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f'{parent_key}{sep}{k}' if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        return f'{cls}_' + '_'.join([f'{key}={value}' for key, value in flatten_dict(dictionary).items()])
        
    def _get_store_path(self,value):
        return Context.get_by_pkvs(self.conf, "store_paths", "name",value,"address")
            
    def __create_storages(self):
        for store_path in self.conf['store_paths']:
            if not os.path.exists(store_path['address']):
                os.mkdir(store_path['address'])
                
    @property                
    def raw_conf(self):
        return self.conf
                
    @property
    def evaluation_metrics(self):
        return self.conf['evaluation_metrics']
        
    @property
    def dataset_store_path(self):
        return self._get_store_path(inspect.stack()[0][3])
    
    @property
    def embedder_store_path(self):
        return self._get_store_path(inspect.stack()[0][3])
    
    @property
    def oracle_store_path(self):
        return self._get_store_path(inspect.stack()[0][3])
    
    @property
    def explainer_store_path(self):
        return self._get_store_path(inspect.stack()[0][3])
    
    @property
    def output_store_path(self):
        return self._get_store_path(inspect.stack()[0][3])
    
    @property
    def log_store_path(self):
        return self._get_store_path(inspect.stack()[0][3])
    
#context = Context.get_context()
#context = Context.get_context("config/test/temp.json")
#print(context)
#print(context._scope)
#print(context.dataset_store_path)


            