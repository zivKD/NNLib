from BL.BaseClasses.CostFunction import CostFunction
import numpy as np

class Quadratic(CostFunction):
    def __init__(self, activationFunction):
        super().__init__(activationFunction)

    def function(self, a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    def derivative(self, z, a, y):
       return np.subtract(a, y) * self.activationFunction.derivative(z)
