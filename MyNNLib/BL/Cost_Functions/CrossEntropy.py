from UI.CostFunction import CostFunction
import numpy as np


class CrossEntropy(CostFunction):
    def function(self, a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    def delta(self, z, a, y):
        return (a - y)