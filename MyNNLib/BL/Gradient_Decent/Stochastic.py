from BL.BaseClasses.GradientDescent import GradientDescent
import numpy as np

class Stochastic(GradientDescent):
    def changeWeights(self, w, gradient, learningRate, mini_batch_size):
        return np.subtract(w, (learningRate / mini_batch_size) * gradient)

    def changeBiases(self, b, gradient, learningRate, mini_batch_size):
        return np.subtract(b, (learningRate / mini_batch_size) * gradient)