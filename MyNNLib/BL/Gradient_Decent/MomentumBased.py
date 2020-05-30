from BL.BaseClasses.GradientDescent import GradientDescent
import numpy as np

from BL.HyperParameterContainer import HyperParameterContainer


class MomentumBased(GradientDescent):
    def __init__(self, numberOfLayersWithWAndB, friction=0.9):
        super().__init__()
        self.__velocities = {
            number : [[],[]] for number in numberOfLayersWithWAndB
        }
        self.__friction = friction

    def setVelocityMatrix(self, numberOfLayer, wShape, bShape):
        layersVelocity = self.__velocities.get(numberOfLayer)
        if layersVelocity != None:
            layersVelocity[0] = np.random.normal(
                    loc=0,
                    scale=np.sqrt(1),
                    size=(wShape)
            )
            layersVelocity[1] = np.random.normal(
                    loc=0,
                    scale=np.sqrt(1),
                    size=(bShape)
            )

    def changeWeights(self, w, gradient, layerNumber):
        vW = self.__velocities.get(layerNumber)
        if vW != None:
            vW = vW[0]
            velocity = self.__friction * vW
            newGradient = HyperParameterContainer.learningRate/HyperParameterContainer.mini_batch_size * gradient
            newVW = np.subtract(velocity, newGradient)
            self.__velocities[layerNumber][1] = newVW
            return np.add(w, newVW)

        return None

    def changeBiases(self, b, gradient, layerNumber):
        vB = self.__velocities.get(layerNumber)
        if vB != None:
            vB = vB[1]
            velocity = self.__friction * vB
            newGradient = HyperParameterContainer.learningRate/HyperParameterContainer.mini_batch_size * gradient
            newVB = np.subtract(velocity, newGradient)
            self.__velocities[layerNumber][1] = newVB
            return np.add(b, newVB)
