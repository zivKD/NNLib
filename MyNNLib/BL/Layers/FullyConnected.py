import numpy as np

from BL.Activation_Functions.Sigmoid import Sigmoid
from BL.BaseClasses.Layer import Layer


class FullyConnected(Layer) :
    number = 0
    def __init__(self,n_in, activationFunction = Sigmoid(), n_out = 100):
        super().__init__(activationFunction, "FullyConnected")
        self.__n_in = n_in
        self.__n_out = n_out
        self._weights = np.random.normal(
            loc=0,
            scale=np.sqrt(1/n_out),
            size=(n_in, n_out)
        )
        self._biases = np.random.normal(
            loc=0,
            scale=1,
            size = (n_out,)
        )

    def feedforward(self, inputs):
        z = np.add(np.dot(self._weights, inputs), self._biases)
        activation = self._activationFunction.function(z)
        self._current_input = inputs
        self._current_weighted_input = z
        self._current_activation = activation
        return activation

    def backpropagate(self, error, learningRate, mini_batch_size, gradient_descent):
        nextError = np.multiply(np.dot(self._weights, error), self._activationFunction.derivative(self._current_weighted_input))
        gradient_descent.changeBiases(self._biases, error, learningRate, mini_batch_size)
        gradient_descent.changeWeights(self._weights, error, learningRate, mini_batch_size)
        return nextError



