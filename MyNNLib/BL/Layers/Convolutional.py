from BL.Activation_Functions.Sigmoid import Sigmoid
from BL.BaseClasses.Layer import Layer
import numpy as np

from BL.Gradient_Decent.Stochastic import Stochastic


class Convolutional(Layer):
    def __init__(self,
                 activationFunction = Sigmoid(),
                 gradientDecent = Stochastic(),
                 sizeOfLocalReceptiveField = (2, 2),
                 stride = 1,
                 numberOfInputFeatureMaps = 1,
                 numberOfFilters = 1,
                 sizeOfInputImage = (5, 5)):
        super().__init__(gradientDecent, activationFunction, "CONVOLUTIONAL")
        self.sizeOfLocalReceptiveField = sizeOfLocalReceptiveField
        self.stride = stride
        self.numberOfInputFeatureMaps = numberOfInputFeatureMaps
        self.sizeOfInputImage = sizeOfInputImage
        self.numberOfFilters = numberOfFilters
        self.initializeFilters()

    def initializeFilters(self):
        counter1 = 0
        counter2 = 0
        for i in range(0, self.sizeOfInputImage[0] - self.sizeOfLocalReceptiveField[0] + 1, self.stride):
            counter1+=1
        for j in range(0, self.sizeOfInputImage[1] - self.sizeOfLocalReceptiveField[1] + 1, self.stride):
            counter2 += 1
        self.numberOfLocalReceptiveFields = max(counter1, counter2) * self.numberOfInputFeatureMaps
        dimensions = (self.numberOfFilters, self.sizeOfLocalReceptiveField[0],
                  self.sizeOfLocalReceptiveField[1])
        self.biases = np.random.normal(
            loc = 0,
            scale = 1,
            size = (self.numberOfFilters,)
        )
        self.biases = np.array([[[[
                bias
                for x in range(self.sizeOfLocalReceptiveField[1])]
                for x in range(self.sizeOfLocalReceptiveField[0])]
                for x in range(self.numberOfLocalReceptiveFields)]
                for bias in self.biases]
        )
        self.weights = np.random.normal(
            loc=0,
            scale=1/np.sqrt(self.numberOfFilters * np.prod(self.sizeOfLocalReceptiveField[0:])),
            size= dimensions
        )
        self.weights = np.array(
            [[self.weights[i] for x in range(self.numberOfLocalReceptiveFields)] for i in range(len(self.weights))]
        )

    def feedforward(self, inputs):
        inputMatrix = self.turnIntoInputMatrix(inputs)
        inputMatrix = [inputMatrix for x in range(self.numberOfFilters)]
        self.currentInputMatrix = np.array(inputMatrix)
        self.currentWeightedInput = np.add(
            self.biases, self.convulotion(self.currentInputMatrix, self.weights))
        self.currentActivation = self.activationFunction.function(self.currentWeightedInput)
        return self.currentActivation

    def backpropagate(self, error):
        flippedWeights = np.array([[
            list(zip(*self.weights[i][j][::-1]))
            for j in range(self.numberOfLocalReceptiveFields)]
            for i in range(len(self.weights))])
        thisLayerError = np.multiply(
            self.convulotion(flippedWeights, error),
            self.activationFunction.derivative(self.currentWeightedInput)
        )
        self.biases = self.gradientDecent.changeBiases(self.biases, error)
        self.weights = self.gradientDecent.changeWeights(self.weights, self.convulotion(self.currentInputMatrix, error))
        return thisLayerError

    def turnIntoInputMatrix(self, inputs):
        inputMatrix = []
        for input in inputs:
            for inputRow in range(0, self.sizeOfInputImage[0] - 1, self.stride) :
                for inputColumn in range(0, self.sizeOfInputImage[1] - 1, self.stride):
                    inputMatrix.append(self.getLocalReceptiveField(inputRow, inputColumn, input))

        return inputMatrix

    def getLocalReceptiveField(self, xStart, yStart, block):
        localReceptiveField = [[None for unit in range(self.sizeOfLocalReceptiveField[0])] for unit2 in range(self.sizeOfLocalReceptiveField[1])]
        for i in range(self.sizeOfLocalReceptiveField[0]):
            for j in range(self.sizeOfLocalReceptiveField[1]):
                localReceptiveField[i][j] = block[xStart + i][yStart+ j]
        return localReceptiveField

    def convulotion(self, matrix, kernel):
        output = np.zeros(matrix.shape)
        kl = len(kernel)
        ks = (kl - 1) // 2  ## kernels usually square with odd number of rows/columns
        imx = len(matrix)
        imy = len(matrix[0])
        for i in range(imx - 1):
            for j in range(imy - 1):
                acc = 0
                for ki in range(kl):  ##kernel is the matrix to be used
                    for kj in range(kl):
                        if 0 <= i - ks <= kl:  ## make sure you don't get out of bound error
                            acc = acc + (matrix[i - ks + ki][j - ks + kj] * kernel[ki][kj])

                output[i][j] = acc

        return output