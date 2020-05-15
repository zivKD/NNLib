from BL.BaseClasses.GradientDecent import GradientDecent
import numpy as np

class Stochastic(GradientDecent):
    def __init__(self, learningRate = 0.1, numberOfMiniBatches = 10):
        super().__init__()
        self.learningRate = learningRate
        self.numberOfMiniBatches = numberOfMiniBatches
        self.coefficent = self.learningRate / self.numberOfMiniBatches


    def changeWeights(self, w, gradient):
        return np.subtract(w, self.coefficent * gradient)

    def changeBiases(self, b, gradient):
        return np.subtract(b, self.coefficent * gradient)