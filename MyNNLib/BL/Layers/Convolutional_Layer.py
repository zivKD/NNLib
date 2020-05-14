from BL.InvalidInputException import InvalidInputException
from UI.Layer import Layer
import numpy as np


class Convolutional_Layer(Layer):
    def __init__(self, number, costFunction, activationFunction,
                 sizeOfLocalReceptiveField = (2, 2),
                 stride = 1,
                 numberOfInputFeatureMaps = 1,
                 numberOfFilters = 1,
                 sizeOfInputImage = (5, 5)):
        super().__init__(number, "CONVOLUTIONAL", costFunction, activationFunction)
        self.sizeOfLocalReceptiveField = sizeOfLocalReceptiveField
        self.stride = stride
        self.numberOfInputFeatureMaps = numberOfInputFeatureMaps
        self.sizeOfInputImage = sizeOfInputImage
        self.numberOfFilters = numberOfFilters
        self.initializeFilters()

    def initializeFilters(self):
        dimensions = (self.numberOfFilters, self.sizeOfLocalReceptiveField[0],
                  self.sizeOfLocalReceptiveField[1])
        self.biases = np.random.normal(
            loc = 0,
            scale = 1,
            size = (self.numberOfFilters,)
        )
        self.weights = np.random.normal(
            loc=0,
            scale=1/np.sqrt(self.numberOfFilters * np.prod(self.sizeOfLocalReceptiveField[0:])),
            size= dimensions
        )

    def feedforward(self, inputs):
        inputMatrix = self.turnIntoInputMatrix(inputs)
        weightMatrix = [[self.weights[i] for numberOfLocalReceptiveFields in inputMatrix] for i in range(len(self.weights))]
        self.currentWeightMatrix = weightMatrix
        biasMatrix = np.array([[[[
                bias
                for x in range(self.sizeOfLocalReceptiveField[1])]
                for x in range(self.sizeOfLocalReceptiveField[0])]
                for x in inputMatrix]
                for bias in self.biases]
        )
        self.currentBiasMatrix = biasMatrix
        inputMatrix = [inputMatrix for x in range(self.numberOfFilters)]
        self.currentInput = inputMatrix
        self.currentActivation = self.activationFunction.function(np.add(biasMatrix, np.matmul(inputMatrix, weightMatrix)))
        return self.currentActivation

    def backpropagate(self, error):
        pass

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