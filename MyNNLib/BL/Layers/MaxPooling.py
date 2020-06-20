from BL.BaseClasses.Layer import Layer
import numpy as np

from BL.HyperParameterContainer import HyperParameterContainer
from BL.Layers.MathHelper import _MathHelper
from DAL.BaseDB import BaseDB


class MaxPooling(Layer):
    def __init__(self, input_image_dims, poolSize, stride, number_of_input_feature_maps):
        super().__init__(layerType="MaxPooling")
        self.__size_of_input_image =input_image_dims
        self.__pool_size = poolSize
        self.__stride = stride
        self.__number_of_input_feature_maps = number_of_input_feature_maps

    def feedforward(self, inputs):
        self._current_input = inputs
        outputImageWidth, outputImageHeight = [x//2 for x in inputs.shape[-2:]]
        localReceptiveFieldedInput = _MathHelper.get_local_receptive_fields(inputs, self.__stride, (outputImageWidth, outputImageHeight), self.__pool_size)
        self.__max = np.amax(localReceptiveFieldedInput, axis=(-2,-1))
        self._current_weighted_input = self.__max.reshape((
            HyperParameterContainer.mini_batch_size,
            self.__number_of_input_feature_maps,
            outputImageHeight,
            outputImageWidth
        ))
        self._current_activation = self._activationFunction.function(self._current_weighted_input)
        return self._current_activation

    def backpropagate(self, error):
        max = self.__max.repeat(2, axis=-1).repeat(2, axis=-2)
        maxCoordinatesGradient = np.equal(self._current_input, max).astype(int)
        error = error.repeat(2, axis=-1).repeat(2, axis=-2).reshape(maxCoordinatesGradient.shape)
        return np.multiply(error, maxCoordinatesGradient)


    def saveToDb(self, db : BaseDB, networkId):
        pass

    def getFromDb(self, db : BaseDB, networkId):
        pass