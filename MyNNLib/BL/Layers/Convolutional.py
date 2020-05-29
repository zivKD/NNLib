from BL.BaseClasses.Layer import Layer
import numpy as np
from BL.HyperParameterContainer import HyperParameterContainer
from BL.Layers.MathHelper import _MathHelper
from DAL.BaseDB import BaseDB

class Convolutional(Layer):
    def __init__(self,
                 sizeOfLocalReceptiveField = (2, 2),
                 stride = 1,
                 numberOfInputFeatureMaps = 1,
                 numberOfFilters = 1,
                 sizeOfInputImage = (5, 5)):
        super().__init__("CONVOLUTIONAL")
        self.__sizeOfLocalReceptiveField = sizeOfLocalReceptiveField
        self.__stride = stride
        self.__numberOfInputFeatureMaps = numberOfInputFeatureMaps
        self.__sizeOfInputImage = sizeOfInputImage
        self.__numberOfFilters = numberOfFilters
        self.__mathHelper = _MathHelper()
        self.__initializeFilters()

    def __initializeFilters(self):
        mini_batch_size = HyperParameterContainer.mini_batch_size
        self.__numberOfLocalReceptiveFields = _MathHelper.getNumberOfLocalReceptiveFields(
            self.__sizeOfInputImage[0],
            self.__sizeOfInputImage[1],
            self.__sizeOfLocalReceptiveField[0],
            self.__sizeOfLocalReceptiveField[1],
            self.__stride,
            self.__numberOfInputFeatureMaps,
            self.__numberOfFilters
        )

        self.__outputImageDims = _MathHelper.getOutputImageDims(
            self.__sizeOfInputImage[0],
            self.__sizeOfInputImage[1],
            self.__sizeOfLocalReceptiveField[0],
            self.__sizeOfLocalReceptiveField[1],
            self.__stride
        )

        biases = np.random.normal(
            loc = 0,
            scale = 1,
            size = (self.__numberOfFilters)
        )
        biases = np.repeat(biases[None, :], mini_batch_size, axis=0)
        self._biases = np.repeat(biases[:, :, None], self.__numberOfLocalReceptiveFields, axis=2)
        weights = np.random.normal(
            loc=0,
            scale=np.sqrt(1/(self.__numberOfFilters * np.prod(self.__sizeOfLocalReceptiveField[0:]))),
            size= (
                self.__numberOfFilters,
                self.__sizeOfLocalReceptiveField[0],
                self.__sizeOfLocalReceptiveField[1])
        )
        self._weights = np.repeat(weights[None, :, :, :], mini_batch_size, axis=0)

    def feedforward(self, inputs):
        inputs = inputs.reshape(
            HyperParameterContainer.mini_batch_size,
            self.__numberOfInputFeatureMaps,
            self.__sizeOfInputImage[0],
            self.__sizeOfInputImage[1]
        )
        inputs = np.repeat(inputs[:,None, :, :, :], self.__numberOfFilters, axis=1)
        self._current_input = inputs
        convolutionProduct = _MathHelper.conv5D(self._current_input, self._weights, self._biases.shape)
        self._current_weighted_input = np.add(self._biases, convolutionProduct)
        self._current_activation = self._activationFunction.function(self._current_weighted_input)
        return self._current_activation

    def backpropagate(self, error):
        flippedWeights = np.array([[
            list(zip(*self._weights[i][j][::-1]))
            for j in range(self.__numberOfLocalReceptiveFields)]
            for i in range(len(self._weights))])

        activation = _MathHelper.conv5D(flippedWeights, error, self._current_weighted_input.shape)
        thisLayerError = np.multiply(
            activation,
            self._activationFunction.derivative(self._current_weighted_input)
        )

        gradient_descent = HyperParameterContainer.gradientDescent
        self._biases = gradient_descent.changeBiases(self._biases, error)
        gradient = _MathHelper.conv5D(self._current_input, error, self._weights)
        self._weights = gradient_descent.changeWeights(
            self._weights,
            gradient
        )

        return thisLayerError

    def saveToDb(self, db : BaseDB, neworkId):
        super().saveToDb(db, neworkId)
        db.saveStride(self.__stride, self.number, neworkId)

    def getFromDb(self, db : BaseDB, networkId):
        super().saveToDb(db, networkId)
        self.__stride = db.getStride(self.number, networkId)

