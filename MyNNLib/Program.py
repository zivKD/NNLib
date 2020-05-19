from BL.Activation_Functions.RELU import RELU
from BL.Cost_Functions.Log_likelihood import Log_likelihood
from BL.Layers.ConvolutionalLayer.Convolutional import Convolutional
from BL.Layers.MaxPooling import MaxPooling
from BL.Network import Network
import numpy as np
# layers = [
#     Convolutional(activationFunction=RELU)
# ]
# net = Network(
#     costFunction=Log_likelihood(),
#     activation_function=RELU(),
#     number_of_epoches=100,
#
# )

arr = np.array([
    [
        [
            [
                [12, 1, 15, 12],
                [13, 13, 13, 13],
                [13, 13, 12, 14]
            ],
            [
                [12, 1, 15, 12],
                [13, 13, 13, 13],
                [13, 13, 12, 14]
            ],
        ],
        [
            [
                [12, 1, 15, 12],
                [13, 13, 13, 13],
                [13, 13, 12, 14]
            ],
            [
                [12, 1, 15, 12],
                [13, 13, 13, 13],
                [13, 13, 12, 14]
            ],
        ],
    ],
    [
        [
            [
                [12, 1, 15, 12],
                [13, 13, 13, 13],
                [13, 13, 12, 14]
            ],
            [
                [12, 1, 15, 12],
                [13, 13, 13, 13],
                [13, 13, 12, 14]
            ],
        ],
        [
            [
                [12, 1, 15, 12],
                [13, 13, 13, 13],
                [13, 13, 12, 14]
            ],
            [
                [12, 1, 15, 12],
                [13, 13, 13, 13],
                [13, 13, 12, 14]
            ],
        ],
    ],
])

layer = MaxPooling()
layer.feedforward(arr)
layer.backpropagate(0, 0, 0, 0)