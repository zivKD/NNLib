from BL.Activation_Functions.Sigmoid import Sigmoid
from BL.BaseClasses.Layer import Layer
import numpy as np

from BL.Layers.MathHelper import _MathHelper
from DAL.BaseDB import BaseDB


class Convolutional(Layer):
    def __init__(self,
                 activationFunction = Sigmoid(),
                 sizeOfLocalReceptiveField = (2, 2),
                 stride = 1,
                 numberOfInputFeatureMaps = 1,
                 numberOfFilters = 1,
                 sizeOfInputImage = (5, 5)):
        super().__init__(activationFunction, "CONVOLUTIONAL")
        self.__sizeOfLocalReceptiveField = sizeOfLocalReceptiveField
        self.__stride = stride
        self.__numberOfInputFeatureMaps = numberOfInputFeatureMaps
        self.__sizeOfInputImage = sizeOfInputImage
        self.__numberOfFilters = numberOfFilters
        self.__mathHelper = _MathHelper()
        [self._weights, self._biases, self.__numberOfLocalReceptiveFields] = self.__mathHelper.initializeFilters(
            self.__sizeOfInputImage,
            self.__sizeOfLocalReceptiveField,
            self.__stride,
            self.__numberOfFilters,
            self.__numberOfInputFeatureMaps
        )

    def feedforward(self, inputs, mini_batch_size):
        inputs = inputs.reshape(mini_batch_size, self.__numberOfInputFeatureMaps, self.__sizeOfInputImage[0],
                     self.__sizeOfInputImage[1])
        inputMatrix = self.__mathHelper.turnIntoInputMatrix(inputs, self.__sizeOfInputImage,
                                                            self.__stride, self.__sizeOfLocalReceptiveField)
        inputMatrix = [inputMatrix for x in range(self.__numberOfFilters)]
        self._current_input = np.array(inputMatrix)
        self._current_weighted_input = np.add(
            self._biases, self.__mathHelper.convulotion(self._current_input, self._weights))

        self._current_activation = self._activationFunction.function(self._current_weighted_input)
        return self._current_activation

    def backpropagate(self, error, learningRate, mini_batch_size, gradient_descent):
        flippedWeights = np.array([[
            list(zip(*self._weights[i][j][::-1]))
            for j in range(self.__numberOfLocalReceptiveFields)]
            for i in range(len(self._weights))])

        thisLayerError = np.multiply(
            self.__mathHelper.convulotion(flippedWeights, error),
            self._activationFunction.derivative(self._current_weighted_input)
        )

        self._biases = gradient_descent.changeBiases(self._biases, error, learningRate, mini_batch_size)
        self._weights = gradient_descent.changeWeights(
            self._weights,
            self.__mathHelper.convulotion(self._current_input, error),
            learningRate,
            mini_batch_size
        )

        return thisLayerError

    def saveToDb(self, db : BaseDB, neworkId):
        super().saveToDb(db, neworkId)
        db.saveStride(self.__stride, self.number, neworkId)

    def getFromDb(self, db : BaseDB, networkId):
        super().saveToDb(db, networkId)
        self.__stride = db.getStride(self.number, networkId)