from BL.BaseClasses.ActivationFunction import ActivationFunction
import numpy as np

class RELU(ActivationFunction):
    def function(self, z):
        return np.maximum(0, z)

    def derivative(self, z):
        return np.heaviside(z, 1)