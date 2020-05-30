import numpy as np

from BL.Activation_Functions.Sigmoid import Sigmoid
from BL.Activation_Functions.Softmax import Softmax
from BL.BaseClasses.Layer import Layer
from BL.HyperParameterContainer import HyperParameterContainer


class FullyConnected(Layer) :
    number = 0
    def __init__(self,n_in, n_out = 100, isSoftmax = False):
        super().__init__("FullyConnected")
        self.__n_in = n_in
        self.__n_out = n_out
        if isSoftmax:
            self._activationFunction = Softmax()

        self._weights = np.random.normal(
            loc=0,
            scale=np.sqrt(1/n_out),
            size=(n_in, n_out)
        )
        biases = np.random.normal(
            loc=0,
            scale=1,
            size = (n_out,)
        )
        self._biases = np.repeat(biases[None, :], HyperParameterContainer.mini_batch_size, axis=0)

    def feedforward(self, inputs):
        inputs = inputs.reshape(HyperParameterContainer.mini_batch_size, self.__n_in)
        dots = []
        for input in inputs:
            dot = np.dot(input, self._weights)
            dots.append(dot)
        z = np.add(dots, self._biases)
        activation = self._activationFunction.function(z)
        self._current_input = inputs
        self._current_weighted_input = z
        self._current_activation = activation
        return activation

    def backpropagate(self, error):
        gradient_descent = HyperParameterContainer.gradientDescent
        dot = np.dot(self._weights, error)
        activationDerivative = self._activationFunction.derivative(self._current_weighted_input)
        # is this the right thing to do?
        activationDerivative = np.repeat(activationDerivative[:, :], self.__n_in//self.__n_out, axis=0)
        nextError = np.multiply(activationDerivative, dot)
        self._biases = gradient_descent.changeBiases(self._biases, error, self.number)
        self._weights = gradient_descent.changeWeights(self._weights, error, self.number)
        return nextError



