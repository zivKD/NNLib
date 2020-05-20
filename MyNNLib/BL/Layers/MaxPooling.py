from BL.Activation_Functions.Sigmoid import Sigmoid
from BL.BaseClasses.ActivationFunction import ActivationFunction
from BL.BaseClasses.Layer import Layer
import numpy as np

from DAL.BaseDB import BaseDB


class MaxPooling(Layer):
    def __init__(self, activationFunction : ActivationFunction):
        super().__init__(activationFunction, layerType="MaxPooling")


    def feedforward(self, inputs):
        self._current_input  = inputs
        # all but the size of the local receptive
        self.__currentIndices = np.argmax(inputs, axis=-1)
        maxOuput = np.max(inputs, axis=-1)
        self._current_weighted_input = maxOuput
        self._current_activation = self._activationFunction.function(maxOuput)
        return self._current_activation

    def backpropagate(self, error, learningRate, mini_batch_size, gradient_descent):
        nextError = np.zeros(self._current_input.shape)
        # Generate the indices to each of the firest dimension
        idx = np.indices(self.__currentIndices.shape)
        indices = [
            idx[j].flatten()
            for j in range(len(self.__currentIndices.shape))
        ]
        # Insert the indices for the last dimension to which we want to insert the 1 where it's max value
        indices.insert(nextError.ndim - 1, self.__currentIndices.flatten())
        # Now we got a tuple of indices for each dimension and we can insert to the last dimensions - 1.
        # The indices tuple will be a 2d, the first dimension is the number of dimensions in the input
        # and the second dimension is the indices which are equal in number to the producet of dimensions
        # in the self.__curentIndices.shape
        nextError[tuple(indices)] = 1
        return nextError

    def saveToDb(self, db : BaseDB, networkId):
        pass

    def getFromDb(self, db : BaseDB, networkId):
        pass