from abc import ABC, abstractmethod

class CostFunction(ABC):
    def __init__(self, activationFunction):
        self.activationFunction = activationFunction

    @abstractmethod
    def function(self, a, y):
        pass

    @abstractmethod
    def delta(self, z, a, y):
        pass