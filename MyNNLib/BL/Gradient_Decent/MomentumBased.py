from BL.BaseClasses.GradientDescent import GradientDescent
import numpy as np

from BL.HyperParameterContainer import HyperParameterContainer


class MomentumBased(GradientDescent):
    def __init__(self, numberOfLayers, friction=0.9):
        super().__init__()
        self.__W_velocities = { number+1 : None for number in range(numberOfLayers) }
        self.__B_velocities = { number+1: None for number in range(numberOfLayers) }
        self.__friction = friction

    def initiaze(self, arr):
        return np.random.normal(
            loc=0,
            scale = np.sqrt(1),
            size = arr.shape
        )

    def changeWeights(self, w, gradient, layerNumber):
        vw = self.__W_velocities[layerNumber]
        if vw is None:
            vw = self.initiaze(w)
        vw = self.__friction * vw
        newGradient = HyperParameterContainer.learningRate/HyperParameterContainer.mini_batch_size * gradient
        newVW = np.subtract(vw, newGradient)
        self.__W_velocities[layerNumber] = newVW
        addition = np.add(w, newVW)
        return addition

    def changeBiases(self, b, gradient, layerNumber):
        vb = self.__B_velocities[layerNumber]
        if vb is None:
            vb = self.initiaze(b)
        gradient = gradient.reshape(vb.shape)
        vb = self.__friction * vb
        newGradient = HyperParameterContainer.learningRate/HyperParameterContainer.mini_batch_size * gradient
        newVB = np.subtract(vb, newGradient)
        self.__B_velocities[layerNumber] = newVB
        return np.add(b, newVB)
