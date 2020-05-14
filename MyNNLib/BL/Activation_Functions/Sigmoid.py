from UI.ActivationFunction import ActivationFunction
import numpy as np


class Sigmoid(ActivationFunction):

    def function(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def derivative(self, z):
        return self.function(z) * (1 - self.function(z))