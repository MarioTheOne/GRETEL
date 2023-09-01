from abc import ABCMeta, abstractmethod

class OriginalAbstractclass(metaclass=ABCMeta):
  @abstractmethod
  def sample_method(self):
    pass

class InheritedAbstractClass(OriginalAbstractclass, metaclass=ABCMeta):
  @abstractmethod
  def another_method(self):
      pass

class ConcreteClass(InheritedAbstractClass):
  def sample_method(self):
    pass

  def another_method(self):
    pass
    

ConcreteClass()