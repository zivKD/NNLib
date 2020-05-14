from abc import ABC, abstractmethod

class CostFunction(ABC):

    @abstractmethod
    def function(self, z):
        pass

    @abstractmethod
    def prime(self, z):
        pass