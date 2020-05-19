from BL.Activation_Functions.Softmax import Softmax
from BL.BaseClasses.CostFunction import CostFunction
import numpy as np

class Log_likelihood(CostFunction):
    def __init__(self):
        super().__init__(Softmax())

    def function(self, a, y):
        return - np.log(a)

    def derivative(self, z, a, y):
       return np.subtract(self.activationFunction.function(z), 1)
