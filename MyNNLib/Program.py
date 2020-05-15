import numpy as np

from BL.Layers.Convolutional import Convolutional

layer = Convolutional(learningRate=0.1,
                      sizeOfLocalReceptiveField=(2, 2),
                      stride = 1,
                      numberOfInputFeatureMaps = 2,
                      numberOfFilters=2,
                      sizeOfInputImage = (2, 4))


weightedInputMetrix = layer.feedforward(np.array([np.array([
    [1, 1, 2 ,2],
    [1, 1, 2, 2],
]),np.array([
    [5, 5, 6 ,6],
    [5, 5, 6, 6],
])]))