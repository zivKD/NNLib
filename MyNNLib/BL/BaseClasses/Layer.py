from abc import ABC, abstractmethod
import numpy as np

from BL.Activation_Functions.Sigmoid import Sigmoid
from BL.BaseClasses.GradientDecent import GradientDecent
from BL.Gradient_Decent.Stochastic import Stochastic


class Layer(ABC) :
    number = 0
    def __init__(self,
                 activationFunction = Sigmoid(),
                 layerType = "non"):
        self.number = Layer.number
        Layer.number+=1
        self.layerType = layerType
        self._activationFunction = activationFunction
        self._weights = []
        self._biases = []
        self._current_input = []
        self._current_weighted_input = []
        self._current_activation = []

    @abstractmethod
    def feedforward(self, inputs):
        pass

    @abstractmethod
    def backpropagate(self, error, learningRate, mini_batch_size, gradient_decent : GradientDecent):
        pass

    @abstractmethod
    def saveToDb(self, db):
        pass