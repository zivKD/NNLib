from abc import ABC, abstractmethod
import numpy as np

from BL.Activation_Functions.Sigmoid import Sigmoid
from BL.BaseClasses.GradientDescent import GradientDescent
from BL.Gradient_Decent.Stochastic import Stochastic
from DAL.BaseDB import BaseDB


class Layer(ABC) :
    number = 0
    def __init__(self,
                 activationFunction = Sigmoid(),
                 layerType = "non"):
        self.number = Layer.number
        Layer.number+=1
        self.layerType = layerType
        self._activationFunction = activationFunction
        self._weights = np.array([])
        self._biases = np.array([])
        self._current_input = []
        self._current_weighted_input = []
        self._current_activation = []

    @abstractmethod
    def feedforward(self, inputs):
        pass

    @abstractmethod
    def backpropagate(self, error, learningRate, mini_batch_size, gradient_descent : GradientDescent):
        pass

    def saveToDb(self, db : BaseDB):
        db.saveBiases(self._biases, self.number)
        db.saveWeights(self._weights, self.number)

    def getWeightsShape(self):
        return self._weights.shape

    def geBiasesShape(self):
        return self._biases.shape