import numpy as np

from BL.BaseClasses.Regularization import Regularization
from BL.HyperParameterContainer import HyperParameterContainer


class Dropout(Regularization):
    def __init__(self, number_of_layers_to_affect : tuple, number_of_runs):
        super().__init__(number_of_layers_to_affect)
        self.__layersWeights = [[num, np.array([None]), number_of_runs] for num in number_of_layers_to_affect]
        self.__number_of_runs = number_of_runs
        self.__p = HyperParameterContainer.dropoutPrecentage

    def changeParams(self, w, b, layerNumber):
        layerWeightArray = []
        for weightArray in self.__layersWeights:
            if(weightArray[0] == layerNumber):
                layerWeightArray = weightArray[1:]

        if layerWeightArray == []:
            pass
        else:
            if layerWeightArray[1] == self.__number_of_runs:
                layerWeightArray[1] = 1
                formerWeights = layerWeightArray[0]
                if formerWeights == None:
                    formerWeights = w
                newWeights = formerWeights
                newWeights[w != 0] = w[w != 0]
                pArr = np.random.binomial(1, self.__p, formerWeights.shape)
                layerWeightArray[0] = newWeights
                w = np.multiply(newWeights, pArr)
            else:
                layerWeightArray[1] += 1