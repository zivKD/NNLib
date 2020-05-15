from abc import ABC, abstractmethod

class GradientDecent(ABC):
    def __init__(self, learningRate=0.1, sizeOfMiniBatch=10):
        self.learningRate = learningRate
        self.numberOfMiniBatches = sizeOfMiniBatch
        self.coefficent = self.learningRate / self.numberOfMiniBatches

    @abstractmethod
    def changeWeights(self, w, gradient):
        pass

    @abstractmethod
    def changeBiases(self, b, gradient):
        pass