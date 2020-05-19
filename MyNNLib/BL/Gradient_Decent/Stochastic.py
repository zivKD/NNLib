from BL.BaseClasses.GradientDecent import GradientDecent
import numpy as np

class Stochastic(GradientDecent):
    def changeWeights(self, w, gradient, learningRate, mini_batch_size):
        return np.subtract(w, (learningRate / mini_batch_size) * gradient)

    def changeBiases(self, b, gradient, learningRate, mini_batch_size):
        return np.subtract(b, (learningRate / mini_batch_size) * gradient)