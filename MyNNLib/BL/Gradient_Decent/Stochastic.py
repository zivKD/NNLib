from BL.BaseClasses.GradientDecent import GradientDecent
import numpy as np

class Stochastic(GradientDecent):
    def __init__(self, learningRate = 0.1, sizeOfMiniBatch = 10):
        super().__init__(learningRate, sizeOfMiniBatch)

    def changeWeights(self, w, gradient):
        return np.subtract(w, self.coefficent * gradient)

    def changeBiases(self, b, gradient):
        return np.subtract(b, self.coefficent * gradient)