import numpy as np

from BL.Activation_Functions.Sigmoid import Sigmoid
from BL.Cost_Functions.CrossEntropy import CrossEntropy
from BL.Layers.Convolutional_Layer import Convolutional_Layer

sigmoid = Sigmoid()
crossEntropy = CrossEntropy(sigmoid)
layer = Convolutional_Layer(0, sigmoid, crossEntropy,
                            sizeOfLocalReceptiveField=(2, 2),
                            stride = 1,
                            numberOfInputFeatureMaps = 2,
                            numberOfFilters=2,
                            sizeOfInputImage = (2, 4))


weightedInputMetrix = layer.feedforward([np.array([
    [1, 1, 2 ,2],
    [1, 1, 2, 2],
]),np.array([
    [5, 5, 6 ,6],
    [5, 5, 6, 6],
])])
