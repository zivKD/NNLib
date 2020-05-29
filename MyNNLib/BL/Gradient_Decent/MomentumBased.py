from BL.BaseClasses.GradientDescent import GradientDescent
import numpy as np

from BL.HyperParameterContainer import HyperParameterContainer


class MomentumBased(GradientDescent):
    def __init__(self, numberOfLayersWithWAndB, friction=0.9):
        super().__init__()
        self.__velocities = np.array([[number, [], []] for number in numberOfLayersWithWAndB])
        self.__numberOfLayers = len(numberOfLayersWithWAndB)
        self.__layerNumW = 0
        self.__layerNumB = 0
        self.__friction = friction

    def setVelocityMatrix(self, numberOfLayer, wShape, bShape):
        for layersVelocity in self.__velocities:
           if(layersVelocity[0] == numberOfLayer):
                layersVelocity[1] = np.random.normal(
                    loc=0,
                    scale=np.sqrt(1),
                    size=(wShape)
                )
                layersVelocity[2] = np.random.normal(
                    loc=0,
                    scale=np.sqrt(1),
                    size=(bShape)
                )
                break

    def changeWeights(self, w, gradient):
        vW = self.__velocities[self.__layerNumW][0]
        newVW = np.subtract(np.dot(vW, self.__friction), np.dot(gradient,
                                                                HyperParameterContainer.learningRate /
                                                                HyperParameterContainer.mini_batch_size))
        self.__velocities[self.__layerNumW][1] = newVW
        self.__layerNumW += 1
        if (self.__layerNumW == self.__numberOfLayers):
            self.__layerNumW = 0
        return np.add(w, newVW)

    def changeBiases(self, b, gradient):
        vB = self.__velocities[self.__layerNumW][1]
        newVB = np.subtract(np.dot(vB, self.__friction), np.dot(gradient,
                                                                HyperParameterContainer.learningRate /
                                                                HyperParameterContainer.mini_batch_size))
        self.__velocities[self.__layerNumW][1] = newVB
        self.__layerNumB += 1
        if (self.__layerNumB == self.__numberOfLayers):
            self.__layerNumB = 0
        return np.add(b, newVB)
