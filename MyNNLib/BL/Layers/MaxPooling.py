from BL.BaseClasses.Layer import Layer
import numpy as np

from BL.HyperParameterContainer import HyperParameterContainer
from BL.Layers.MathHelper import _MathHelper
from DAL.BaseDB import BaseDB


class MaxPooling(Layer):
    def __init__(self, sizeOfInputImage, poolSize, stride, number_of_input_feature_maps):
        super().__init__(layerType="MaxPooling")
        self.__size_of_input_image =sizeOfInputImage
        self.__pool_size = poolSize
        self.__stride = stride
        self.__number_of_input_feature_maps = number_of_input_feature_maps

    def feedforward(self, inputs):
        helper = _MathHelper()
        numberOfFilters = len(inputs[1])
        numberOfLocalReceptiveFields = helper.getNumberOfLocalReceptiveFields(
            self.__size_of_input_image, self.__pool_size, self.__stride, self.__number_of_input_feature_maps
        )
        inputMatrix = helper.turnIntoInputMatrix2(inputs,
                                                  self.__size_of_input_image, self.__stride, self.__pool_size,
                                                 self.__number_of_input_feature_maps)
        self._current_input = np.array(inputMatrix)
        # all but the size of the local receptive
        self.__currentIndices = np.argmax(self._current_input, axis=-1)
        maxOuput = np.max(self._current_input, axis=-1)
        self._current_weighted_input = maxOuput
        self._current_activation = self._activationFunction.function(maxOuput)
        return helper.turnIntoCommonShape(self._current_activation, numberOfFilters,
                                          numberOfLocalReceptiveFields)

    def backpropagate(self, error):
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