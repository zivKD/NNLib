import numpy as np

from BL.BaseClasses.Regularization import Regularization
from BL.HyperParameterContainer import HyperParameterContainer


class Dropout(Regularization):
    def __init__(self, number_of_layers_to_affect : tuple, number_of_runs):
        super().__init__(number_of_layers_to_affect)
        self.__layersWeights = {num : {"weights" : None, "number_of_runs" : number_of_runs} for num in number_of_layers_to_affect
        }
        self.__number_of_runs = number_of_runs
        self.__p = HyperParameterContainer.dropoutPercentage

    def changeParams(self, w, b, layerNumber):
        matrix = self.__layersWeights.get(layerNumber)
        if matrix is None == False:
            if matrix['number_of_runs'] == self.__number_of_runs:
                matrix['number_of_runs'] = 0
                formerWeights = matrix['weights']
                if formerWeights is None:
                    formerWeights = w
                newWeights = formerWeights
                newWeights[w != 0] = w[w != 0]
                pArr = np.random.binomial(1, self.__p, formerWeights.shape)
                matrix['weights'] = newWeights
                w = np.multiply(newWeights, pArr)
            else:
                matrix['number_of_runs'] += 1

        return (w, b)