import numpy as np

from BL.BaseClasses.Regularization import Regularization


class Dropout(Regularization):
    def __init__(self, numberOfLayers, number_of_runs, p = 0.5):
        self.__layerWeights = [[np.array([None]), 1] for i in range(numberOfLayers)]
        self.__number_of_runs = number_of_runs
        self.__p = p

    def changeParams(self, w, b, layerNumber):
        if self.__layerWeights[layerNumber][1] == self.__number_of_runs:
            self.__layerWeights[layerNumber][1] = 0
            formerWeights = self.__layerWeights[layerNumber][0]
            if formerWeights == None:
                formerWeights = w
            newWeights = formerWeights
            newWeights[w != 0] = w[w != 0]
            pArr = np.random.binomial(1, self.__p, formerWeights.shape)
            self.__layerWeights[layerNumber][0] = newWeights
            w = np.multiply(newWeights, pArr)
        else:
            self.__layerWeights[layerNumber][1] += 1