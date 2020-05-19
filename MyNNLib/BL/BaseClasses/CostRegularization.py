from abc import ABC, abstractmethod

class CostRegularization(ABC):
    def __init__(self, regularizationParameter):
        self.regularizationParameter = regularizationParameter

    @abstractmethod
    def regularizationTerm(self):
        pass

    @abstractmethod
    def changeWeights(self, w):
        pass

    @abstractmethod
    def changeBiases(self, b):
        pass