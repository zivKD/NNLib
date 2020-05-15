from BL.BaseClasses.ActivationFunction import ActivationFunction
import numpy as np

class Softmax(ActivationFunction):
    def function(self, z):
        e_x = np.exp(z - np.max(z))
        return e_x / e_x.sum()

    def derivative(self, z):
        return 1 - self.function(z)