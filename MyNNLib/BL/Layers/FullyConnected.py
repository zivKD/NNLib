import numpy as np
from BL.Activation_Functions.Softmax import Softmax
from BL.BaseClasses.Layer import Layer
from BL.HyperParameterContainer import HyperParameterContainer
from BL.Layers.MathHelper import _MathHelper


class FullyConnected(Layer) :
    number = 0
    def __init__(self,n_in, n_out = 100, isSoftmax = False):
        super().__init__("FullyConnected")
        self.__n_in = n_in
        self.__n_out = n_out
        self.__isSoftmax = isSoftmax
        if isSoftmax:
            self._activationFunction = Softmax()
        weights = np.random.normal(loc=0, scale=np.sqrt(1/n_out), size=(n_in, n_out))
        self._weights = _MathHelper.repeat(weights, axis=0, num_of_repeats=HyperParameterContainer.mini_batch_size)
        biases = np.random.normal(loc=0, scale=1, size = (n_out,))
        self._biases = _MathHelper.repeat(biases, axis=0, num_of_repeats=HyperParameterContainer.mini_batch_size)

    def feedforward(self, inputs):
        inputs = inputs.reshape(HyperParameterContainer.mini_batch_size, self.__n_in)
        dots = []
        for i in range(len(inputs)):
            dots.append(np.dot(inputs[i], self._weights[i]))
        z = np.add(dots, self._biases)
        activation = self._activationFunction.function(z)
        self._current_input = inputs
        self._current_weighted_input = z
        self._current_activation = activation
        return activation

    def backpropagate(self, error):
        self.change_by_gradient(error)
        thisLayerError = self.calculate_this_layer_error(error)
        return thisLayerError

    def calculate_this_layer_error(self, error):
        weights = self._weights.transpose()
        weights = weights.reshape(self._weights.shape)
        activationDerivative = self._activationFunction.derivative(self._current_weighted_input)
        activationDerivative = activationDerivative.repeat(self.__n_in, axis=1)
        dots = []
        for i in range(len(error)):
            dots.append(np.dot(weights[i], error[i]))
        dots = _MathHelper.repeat(dots, axis=1, num_of_repeats=self.__n_out, should_expand=False)
        nextError = np.multiply(dots, activationDerivative)
        nextError = nextError.reshape(HyperParameterContainer.mini_batch_size, self.__n_in, self.__n_out)
        return np.average(nextError, axis=-1).reshape(HyperParameterContainer.mini_batch_size, self.__n_in)

    def change_by_gradient(self, error):
        wGradient = np.dot(error, self._current_activation.transpose())
        num_of_repeats = (HyperParameterContainer.mini_batch_size,
                          self._weights.shape[1] / wGradient.shape[0], self._weights.shape[2] / wGradient.shape[1])
        wGradient = _MathHelper.repeat(
            wGradient,
            axis=(0,1,2),
            num_of_repeats= num_of_repeats,
            should_expand=(True,False,False)
        )
        self._weights = HyperParameterContainer.gradientDescent.changeWeights(self._weights, wGradient, self.number)
        self._biases = HyperParameterContainer.gradientDescent.changeBiases(self._biases, error, self.number)
