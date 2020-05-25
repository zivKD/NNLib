import numpy as np

from BL.HyperParameterContainer import HyperParameterContainer


class _MathHelper():
    def turnIntoInputMatrix(self, inputs, sizeOfInputImage, stride, sizeOfLocalReceptiveField, numberOfInputFeatureMaps):
        inputMatrix = [[] for i in range(len(inputs))]
        i = 0
        for input in inputs:
            for inputRow in range(0, sizeOfInputImage[0] - sizeOfLocalReceptiveField[0], stride) :
                    for inputColumn in range(0, sizeOfInputImage[1] - sizeOfLocalReceptiveField[1], stride):
                        for j in range(numberOfInputFeatureMaps):
                            inputMatrix[i].append(self.getLocalReceptiveField(inputRow, inputColumn, input[j], sizeOfLocalReceptiveField))
            i+=1
        return inputMatrix

    def getLocalReceptiveField(self, xStart, yStart, block, sizeOfLocalReceptiveField):
        localReceptiveField = [[None for unit in range(sizeOfLocalReceptiveField[0])] for unit2 in range(sizeOfLocalReceptiveField[1])]
        for i in range(sizeOfLocalReceptiveField[0]):
            for j in range(sizeOfLocalReceptiveField[1]):
                localReceptiveField[i][j] = block[xStart + i][yStart+ j]
        return localReceptiveField

    def convulotion(self, matrix, kernel):
        output = np.zeros(matrix.shape)
        kl = len(kernel)
        ks = (kl - 1) // 2  ## kernels usually square with odd number of rows/columns
        imx = len(matrix)
        imy = len(matrix[0])
        for i in range(imx - kl):
            for j in range(imy - kl):
                acc = 0
                for ki in range(kl - 1):  ##kernel is the matrix to be used
                    for kj in range(kl - 1):
                        if 0 <= i - ks <= kl:  ## make sure you don't get out of bound error
                            acc = acc + (matrix[i - ks + ki][j - ks + kj] * kernel[ki][kj])

                output[i][j] = acc

        return output