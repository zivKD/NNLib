from abc import ABC, abstractmethod

class ActivationFunction(ABC):

    @abstractmethod
    def function(self, z):
        pass

    @abstractmethod
    def derivative(self, z):
        pass