import numpy as np

from BL.Activation_Functions.Sigmoid import Sigmoid
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

        weights = np.random.normal(
            loc=0,
            scale=np.sqrt(1/n_out),
            size=(n_in, n_out)
        )
        self._weights = np.repeat(weights[None, :, :], HyperParameterContainer.mini_batch_size, axis=0)
        biases = np.random.normal(
            loc=0,
            scale=1,
            size = (n_out,)
        )
        self._biases = np.repeat(biases[None, :], HyperParameterContainer.mini_batch_size, axis=0)

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
        """
            δ = ((w)^T * δ) ⊙ σ`(z)
        """
        # we transpose the weights because we are moving backwards
        weights = self._weights.transpose()
        weights = weights.reshape(self._weights.shape)
        activationDerivative = self._activationFunction.derivative(self._current_weighted_input)
        nextError = []
        if weights.shape[1] % activationDerivative.shape[1] == 0:
            activationDerivative = activationDerivative.repeat(weights.shape[1]/activationDerivative.shape[1], axis=1)
            dots = []
            for i in range(len(error)):
                dots.append(np.dot(weights[i], error[i]))
            nextError = np.multiply(dots, activationDerivative)
        else:
            activationDerivative = activationDerivative.repeat(self.__n_in, axis=1)
            dots = []
            for i in range(len(error)):
                dots.append(np.dot(weights[i], error[i]))
            dots = np.repeat(dots, self.__n_out, axis=1)
            nextError = np.multiply(dots, activationDerivative)
            nextError = nextError.reshape(HyperParameterContainer.mini_batch_size, self.__n_in, self.__n_out)
            nextError = np.average(nextError, axis=-1).reshape(HyperParameterContainer.mini_batch_size, self.__n_in)
        '''
            ∂C/∂w = a_in * δ_out
        '''
        # we transpose the activation for the same reason we transpose the weights
        wGradient = np.dot(error, self._current_activation.transpose())
        wGradient = np.repeat(wGradient[None, :, :], HyperParameterContainer.mini_batch_size, axis=0)
        wGradient = wGradient.repeat(self._weights.shape[1] / wGradient.shape[1], axis=1)
        wGradient = wGradient.repeat(self._weights.shape[2] / wGradient.shape[2], axis=2)
        self._weights = HyperParameterContainer.gradientDescent.changeWeights(
            self._weights, wGradient, self.number
        )

        '''
            ∂C/∂b = δ
        '''
        bGradient = error
        self._biases = HyperParameterContainer.gradientDescent.changeBiases(
            self._biases, bGradient, self.number
        )

        return nextError

