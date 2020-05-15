from abc import ABC, abstractmethod

class GradientDecent(ABC):
    @abstractmethod
    def changeWeights(self, w, gradient):
        pass

    @abstractmethod
    def changeBiases(self, b, gradient):
        pass