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
        self.__initializeFilters()

    def __initializeFilters(self):
        mini_batch_size = HyperParameterContainer.mini_batch_size
        counter1 = 0
        counter2 = 0
        for i in range(0, self.__sizeOfInputImage[0] - self.__sizeOfLocalReceptiveField[0], self.__stride):
            counter1+=1
        for i in range(0, self.__sizeOfInputImage[1] - self.__sizeOfLocalReceptiveField[1], self.__stride):
            counter2 += 1
        self.__numberOfLocalReceptiveFields = counter1 * counter2 * self.__numberOfInputFeatureMaps
        biases = np.random.normal(
            loc = 0,
            scale = 1,
            size = (HyperParameterContainer.mini_batch_size)
        )
        self._biases = np.array([[[[[
                bias
                for x in range(self.__sizeOfLocalReceptiveField[1])]
                for x in range(self.__sizeOfLocalReceptiveField[0])]
                for x in range(self.__numberOfLocalReceptiveFields)]
                for x in range(self.__numberOfFilters)]
                for bias in biases]
        )
        weights = np.random.normal(
            loc=0,
            scale=np.sqrt(1/(self.__numberOfFilters * np.prod(self.__sizeOfLocalReceptiveField[0:]))),
            size= (
                HyperParameterContainer.mini_batch_size,
                self.__numberOfFilters,
                self.__sizeOfLocalReceptiveField[0],
                self.__sizeOfLocalReceptiveField[1])
        )
        self._weights = np.array(
            [[[
               filter
                for x in range(self.__numberOfLocalReceptiveFields)]
                for filter in weightSlice]
                for weightSlice in weights]
        )

    def feedforward(self, inputs):
        inputs = inputs.reshape(
            HyperParameterContainer.mini_batch_size,
            self.__numberOfInputFeatureMaps,
            self.__sizeOfInputImage[0],
            self.__sizeOfInputImage[1]
        )

        inputMatrix = self.__mathHelper.turnIntoInputMatrix(
            inputs,
            self.__sizeOfInputImage,
            self.__stride,
            self.__sizeOfLocalReceptiveField,
            self.__numberOfInputFeatureMaps
        )

        inputMatrix = [[
            input
            for x in range(self.__numberOfFilters)]
            for input in inputMatrix]

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