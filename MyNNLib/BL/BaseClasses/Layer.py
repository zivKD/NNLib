from abc import ABC, abstractmethod
import numpy as np
from BL.BaseClasses.GradientDescent import GradientDescent
from BL.BaseClasses.Regularization import Regularization
from BL.HyperParameterContainer import HyperParameterContainer
from DAL.BaseDB import BaseDB


class Layer(ABC) :
    number = 1
    def __init__(self,
                 layerType = "non"
                 ):
        self.number = Layer.number
        Layer.number+=1
        self.layerType = layerType
        self._activationFunction = HyperParameterContainer.activationFunction
        self._weights = np.array([])
        self._biases = np.array([])
        self._current_input = []
        self._current_weighted_input = []
        self._current_activation = []

    @abstractmethod
    def feedforward(self, inputs):
        pass

    @abstractmethod
    def backpropagate(self, error):
        pass

    def regulate(self, regularization : Regularization):
        regularization.changeParams(self._weights, self._biases, self.number)

    def softmax(self):
        self._activationFunction.setWeightedInputs(self._current_weighted_input)

    def saveToDb(self, db : BaseDB, networkId):
        db.saveBiases(self._biases, self.number, networkId)
        db.saveWeights(self._weights, self.number, networkId)

    def getFromDb(self, db : BaseDB, networkId):
        self._weights = db.getWeights(self.number, networkId)
        self._biases = db.getBiases(self.number, networkId)

    def getWeightShape(self):
        return self._weights.shape

    def getBiasShape(self):
        return self._biases.shape