from BL.BaseClasses.CostRegularization import CostRegularization
from BL.BaseClasses.GradientDescent import GradientDescent
import numpy as np

class MomemntumBased(GradientDescent):
    def __init__(self, numberOfLayersWithWAndB, friction = 0.9):
        self.__velocities = [[[], []] for i in range(numberOfLayersWithWAndB)]
        self.__numberOfLayers = numberOfLayersWithWAndB
        self.__layerNumW = 0
        self.__layerNumB = 0
        self.__friction = friction

    def setVelocityMatrix(self, numberOfLayer, wShape, bShape):
        self.__velocities[numberOfLayer][0] = np.random.normal(
            loc=0,
            scale=np.sqrt(1),
            size=(wShape)
        )
        self.__velocities[numberOfLayer][1] = np.random.normal(
            loc=0,
            scale=np.sqrt(1),
            size=(bShape)
        )

    def changeWeights(self, w, gradient, learningRate, mini_batch_size):
        vW = self.__velocities[self.__layerNumW][0]
        newVW = np.subtract(np.dot(vW, self.__friction), np.dot(gradient, learningRate/mini_batch_size))
        self.__velocities[self.__layerNumW][1] = newVW
        self.__layerNumW+=1
        if(self.__layerNumW == self.__numberOfLayers):
            self.__layerNumW = 0
        return np.add(w, newVW)

    def changeBiases(self, b, gradient, learningRate, mini_batch_size):
        vB = self.__velocities[self.__layerNumW][1]
        newVB = np.subtract(np.dot(vB, self.__friction), np.dot(gradient, learningRate / mini_batch_size))
        self.__velocities[self.__layerNumW][1] = newVB
        self.__layerNumB += 1
        if (self.__layerNumB == self.__numberOfLayers):
            self.__layerNumB = 0
        return np.add(b, newVB)