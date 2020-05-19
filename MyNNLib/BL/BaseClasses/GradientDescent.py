from abc import ABC, abstractmethod

class GradientDescent(ABC):
    @abstractmethod
    def changeWeights(self, w, gradient, learningRate, mini_batch_size):
        pass

    @abstractmethod
    def changeBiases(self, b, gradient, learningRate, mini_batch_size):
        pass