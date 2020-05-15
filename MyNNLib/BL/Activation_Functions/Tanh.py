from BL.BaseClasses.ActivationFunction import ActivationFunction
import numpy as np

class Tanh(ActivationFunction):
    def function(self, z):
        return np.tanh(z)

    def derivative(self, z):
        return 1 - self.function(z) ** 2