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
        self._current_input = inputs
        outputImageWidth, outputImageHeight = [x//2 for x in inputs.shape[-2:]]
        localReceptiveFieldedInput = _MathHelper.getLocalReceptiveFields(
            inputs, self.__stride, outputImageWidth, outputImageHeight,
            self.__pool_size[0], self.__pool_size[1]
        )

        [max, self.__currentIndices] = self.getMax(localReceptiveFieldedInput)
        self._current_weighted_input = max.reshape((
            HyperParameterContainer.mini_batch_size,
            self.__number_of_input_feature_maps,
            outputImageHeight,
            outputImageWidth
        ))

        self._current_activation = self._activationFunction.function(self._current_weighted_input)
        return self._current_activation

    def backpropagate(self, error):
        nextError = np.zeros(self._current_input.shape)
        # Generate the indices to each of the first dimension
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
        nextError[self.__currentIndices] = 1
        return nextError


    def saveToDb(self, db : BaseDB, networkId):
        pass

    def getFromDb(self, db : BaseDB, networkId):
        pass

    def getMax(self, localReceptiveFieldedInput):
        reshapedrfiInput = localReceptiveFieldedInput.reshape(localReceptiveFieldedInput.shape[:-2] + (-1,))
        ms = np.max(reshapedrfiInput, axis=-1)
        idx = np.argmax(reshapedrfiInput, -1)
        idx1 = np.unravel_index(idx, localReceptiveFieldedInput.shape[-2:])
        ij = np.ix_(*[np.arange(i) for i in localReceptiveFieldedInput.shape[:-2]])
        indices = ij + idx1
        return [ms, indices]
