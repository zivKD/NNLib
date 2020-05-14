from UI.CostFunction import CostFunction
import numpy as np


class Sigmoid(CostFunction):

    def function(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def prime(self, z):
        return self.function(z) * (1 - self.function(z))