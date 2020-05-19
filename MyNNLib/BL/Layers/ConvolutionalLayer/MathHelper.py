import numpy as np

class _MathHelper():
    def initializeFilters(self, sizeOfInputImage, sizeOfLocalReceptiveField, stride, numberOfFilters, numberOfInputFeatureMaps):
        counter1 = 0
        counter2 = 0
        for i in range(0, sizeOfInputImage[0] - sizeOfLocalReceptiveField[0] + 1, stride):
            counter1+=1
        for j in range(0, sizeOfInputImage[1] - sizeOfLocalReceptiveField[1] + 1, stride):
            counter2 += 1
        numberOfLocalReceptiveFields = max(counter1, counter2) * numberOfInputFeatureMaps
        dimensions = (numberOfFilters, sizeOfLocalReceptiveField[0],
                  sizeOfLocalReceptiveField[1])
        biases = np.random.normal(
            loc = 0,
            scale = 1,
            size = (numberOfFilters,)
        )
        biases = np.array([[[[
                bias
                for x in range(sizeOfLocalReceptiveField[1])]
                for x in range(sizeOfLocalReceptiveField[0])]
                for x in range(numberOfLocalReceptiveFields)]
                for bias in biases]
        )
        weights = np.random.normal(
            loc=0,
            scale=np.sqrt(1/(numberOfFilters * np.prod(sizeOfLocalReceptiveField[0:]))),
            size= dimensions
        )
        weights = np.array(
            [[weights[i] for x in range(numberOfLocalReceptiveFields)] for i in range(len(weights))]
        )

        return [weights, biases, numberOfLocalReceptiveFields]

    def turnIntoInputMatrix(self, inputs, sizeOfInputImage, stride, sizeOfLocalReceptiveField):
        inputMatrix = []
        for input in inputs:
            for inputRow in range(0, sizeOfInputImage[0] - 1, stride) :
                for inputColumn in range(0, sizeOfInputImage[1] - 1, stride):
                    inputMatrix.append(self.getLocalReceptiveField(inputRow, inputColumn, input, sizeOfLocalReceptiveField))

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
        for i in range(imx - 1):
            for j in range(imy - 1):
                acc = 0
                for ki in range(kl):  ##kernel is the matrix to be used
                    for kj in range(kl):
                        if 0 <= i - ks <= kl:  ## make sure you don't get out of bound error
                            acc = acc + (matrix[i - ks + ki][j - ks + kj] * kernel[ki][kj])

                output[i][j] = acc

        return output