import numpy as np

from BL.HyperParameterContainer import HyperParameterContainer

class _MathHelper():
    def getNumberOfLocalReceptiveFields(self,sizeOfInputImage, sizeOfLocalReceptiveField, stride, numberOfInputFeatureMaps):
        counter1 = 0
        counter2 = 0
        for i in range(0, sizeOfInputImage[0] - sizeOfLocalReceptiveField[0], stride):
            counter1 += 1
        for i in range(0, sizeOfInputImage[1] - sizeOfLocalReceptiveField[1], stride):
            counter2 += 1
        return counter1 * counter2 * numberOfInputFeatureMaps

    def turnIntoInputMatrix(self, inputs, sizeOfInputImage, stride, sizeOfLocalReceptiveField, numberOfInputFeatureMaps):
        inputMatrix = [[] for i in range(len(inputs))]
        i = 0
        for input in inputs:
            for inputRow in range(0, sizeOfInputImage[0] - sizeOfLocalReceptiveField[0], stride) :
                    for inputColumn in range(0, sizeOfInputImage[1] - sizeOfLocalReceptiveField[1], stride):
                        for j in range(numberOfInputFeatureMaps):
                            inputMatrix[i].append(self.getLocalReceptiveField(inputRow, inputColumn, input[j],
                                                                              sizeOfLocalReceptiveField))
            i+=1
        return inputMatrix

    def turnIntoInputMatrix2(self, inputs, sizeOfInputImage, stride, sizeOfLocalReceptiveField,
                            numberOfInputFeatureMaps):
        inputMatrix = [[[] for j in range(len(inputs[0]))] for i in range(len(inputs))]
        i = 0
        j = 0
        for filter in inputs:
            for input in filter:
                for inputRow in range(0, sizeOfInputImage[0] - sizeOfLocalReceptiveField[0], stride):
                    for inputColumn in range(0, sizeOfInputImage[1] - sizeOfLocalReceptiveField[1], stride):
                        for m in range(numberOfInputFeatureMaps):
                            localReceptiveField = self.getLocalReceptiveField(inputRow, inputColumn, input,
                                      sizeOfLocalReceptiveField)
                            if localReceptiveField is not None:
                                inputMatrix[i][j].append(localReceptiveField)
                j += 1
            j = 0
            i += 1
        return inputMatrix

    def getLocalReceptiveField(self, xStart, yStart, block, sizeOfLocalReceptiveField):
        localReceptiveField = [
            [None for unit in range(sizeOfLocalReceptiveField[0])]
            for unit2 in range(sizeOfLocalReceptiveField[1])]
        for i in range(sizeOfLocalReceptiveField[0]):
            for j in range(sizeOfLocalReceptiveField[1]):
                # if the size of the image isn't dividable by the local receptive size
                # just drop the data
                if(xStart + i < len(block) and yStart + j < len(block[0])):
                    localReceptiveField[i][j] = block[xStart + i][yStart+ j]
                else:
                    return None
        return localReceptiveField

    def turnIntoCommonShape(self, activation, numberOfFilters, numberOfLocalReceptiveFields, sizeOfLocalReceptiveField):
        sqrt = int(np.sqrt(numberOfLocalReceptiveFields))  # based on being a square
        imageWidth = sqrt * sizeOfLocalReceptiveField[0]
        imageHeight = sqrt * sizeOfLocalReceptiveField[1]
        reshapedActivation = activation.reshape(
            HyperParameterContainer.mini_batch_size,
            numberOfFilters,
            imageWidth,
            imageHeight
        )
        return reshapedActivation