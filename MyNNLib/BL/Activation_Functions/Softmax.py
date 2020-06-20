from BL.BaseClasses.ActivationFunction import ActivationFunction
import numpy as np

class Softmax(ActivationFunction):
    def function(self, z):
        maxValue = np.max(z)
        e_x = np.exp(z - maxValue)
        returnValue = e_x / e_x.sum(axis=0)
        return returnValue

    def derivative(self, z):
        return self.function(z) - 1