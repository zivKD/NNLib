from abc import ABC, abstractmethod
import numpy as np

from BL.Activation_Functions.Sigmoid import Sigmoid
from BL.Gradient_Decent.Stochastic import Stochastic


class Layer(ABC) :
    number = 0
    def __init__(self,
                 gradientDecent = Stochastic(),
                 activationFunction = Sigmoid(),
                 layerType = "non"):
        self.number = Layer.number
        Layer.number+=1
        self.gradientDecent = gradientDecent
        self.layerType = layerType
        self.activationFunction = activationFunction
        self.weights = []
        self.biases = []

    @abstractmethod
    def feedforward(self, inputs):
        pass

    @abstractmethod
    def backpropagate(self, error):
        pass