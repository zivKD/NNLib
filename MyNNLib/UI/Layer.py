from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC) :
    def __init__(self, number, costFunction, activationFunction, layerType = "non"):
        self.number = number
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