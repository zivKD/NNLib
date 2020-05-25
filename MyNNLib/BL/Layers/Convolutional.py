from BL.Activation_Functions.Sigmoid import Sigmoid
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
        [self._weights, self._biases, self.__numberOfLocalReceptiveFields] = self.__mathHelper.initializeFilters(
            HyperParameterContainer.mini_batch_size,
            self.__sizeOfInputImage,
            self.__sizeOfLocalReceptiveField,
            self.__stride,
            self.__numberOfFilters,
            self.__numberOfInputFeatureMaps
        )

    def feedforward(self, inputs):
        inputs = inputs.reshape(HyperParameterContainer.mini_batch_size, self.__numberOfInputFeatureMaps, self.__sizeOfInputImage[0],
                     self.__sizeOfInputImage[1])
        inputMatrix = self.__mathHelper.turnIntoInputMatrix(
            inputs,
            self.__sizeOfInputImage,
            self.__stride,
            self.__sizeOfLocalReceptiveField,
            self.__numberOfInputFeatureMaps
        )
        inputMatrix = [inputMatrix for x in range(self.__numberOfFilters)]
        self._current_input = np.array(inputMatrix)
        self._current_weighted_input = np.add(
            self._biases, self.__mathHelper.convulotion(self._current_input, self._weights))

        self._current_activation = self._activationFunction.function(self._current_weighted_input)
        return self._current_activation

    def backpropagate(self, error):
        flippedWeights = np.array([[
            list(zip(*self._weights[i][j][::-1]))
            for j in range(self.__numberOfLocalReceptiveFields)]
            for i in range(len(self._weights))])

        thisLayerError = np.multiply(
            self.__mathHelper.convulotion(flippedWeights, error),
            self._activationFunction.derivative(self._current_weighted_input)
        )

        gradient_descent = HyperParameterContainer.gradientDescent
        self._biases = gradient_descent.changeBiases(self._biases, error)
        self._weights = gradient_descent.changeWeights(
            self._weights,
            self.__mathHelper.convulotion(self._current_input, error),
        )

        return thisLayerError

    def saveToDb(self, db : BaseDB, neworkId):
        super().saveToDb(db, neworkId)
        db.saveStride(self.__stride, self.number, neworkId)

    def getFromDb(self, db : BaseDB, networkId):
        super().saveToDb(db, networkId)
        self.__stride = db.getStride(self.number, networkId)