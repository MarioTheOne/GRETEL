import os
from src.core.grtl_base import Base
from flufl.lock import Lock
from datetime import timedelta
from abc import ABCMeta,abstractmethod

class Savable(Base,metaclass=ABCMeta):
    
    @abstractmethod 
    def write(self):
        pass

    @abstractmethod 
    def read(self):
        pass

    def saved(self):
        return os.path.exists(self.context.get_path(self))
    
    def load_or_create(self, condition=None):
        condition = condition or not self.saved() if condition else not self.saved()
        lock = Lock(self.context.get_path(self)+'.lck',lifetime=timedelta(hours=self.context.lock_release_tout))
        with lock: #TODO: Check if it is possible to move it inside the if TRUE branch below to avoid small locks
            if condition:
                self.context.logger.info(f"Need to be created: {self}")
                self.create()
                self.write()
                self.context.logger.info(str(self)+" saved.")
            else:
                self.context.logger.info(f"Loading: {self}")
                self.read()
                self.context.logger.info(f"Loaded: {self}")
    
    @abstractmethod
    def create(self):
        pass