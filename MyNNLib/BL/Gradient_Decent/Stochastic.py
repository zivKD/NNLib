from BL.BaseClasses.GradientDescent import GradientDescent
import numpy as np

from BL.HyperParameterContainer import HyperParameterContainer


class Stochastic(GradientDescent):
    def changeWeights(self, w, gradient, layerNumber):
        return np.subtract(w, (HyperParameterContainer.learningRate / HyperParameterContainer.mini_batch_size) * gradient)

    def changeBiases(self, b, gradient, layerNumber):
        return np.subtract(b, (HyperParameterContainer.learningRate / HyperParameterContainer.mini_batch_size) * gradient)