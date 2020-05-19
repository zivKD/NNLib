from random import random
from BL.Activation_Functions.Sigmoid import Sigmoid
from BL.Cost_Functions.Quadratic import Quadratic
from BL.Gradient_Decent.Stochastic import Stochastic
from BL.Layers.ConvolutionalLayer.Convolutional import Convolutional
from DAL.Mongo.MongoDB import MongoDB
import numpy as np


class Network():
    def __init__(self,
                 costFunction=Quadratic(Sigmoid()),
                 learningRate = 2,
                 last_layer_activation_function=Sigmoid(),
                 gradient_decent = Stochastic(),
                 mini_batch_size=3,
                 number_of_epoches=5,
                 layers=(Convolutional(),),
                 training_set=(),
                 validation_set=(),
                 test_set=(),
                 db=MongoDB()):
        self.costFunction = costFunction
        self.learningRate = learningRate
        self.last_layer_activation_function = last_layer_activation_function
        self.gradient_decent = gradient_decent
        self.mini_batch_size = mini_batch_size
        self.number_of_epochs = number_of_epoches
        self.layers = layers
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.db = db

    def runNetwork(self, onMonitoring=lambda accuracy: print(accuracy), frequencyOfMonitoring=10):
        for i in range(self.number_of_epochs):
            n = len(self.training_set)
            random.shuffle(self.training_set)
            mini_batches = [
                self.training_set[k:k + self.mini_batch_size]
                for k in range(0, n, self.mini_batch_size)
            ]

            monitoring_counter = 0
            for mini_batch in mini_batches:
                monitoring_counter += 1
                x = np.array([batch[0].ravel() for batch in mini_batch]).transpose()
                y = np.array([batch[1].ravel() for batch in mini_batch]).transpose()
                output = self.__feedForward(x, y)
                if (monitoring_counter == frequencyOfMonitoring):
                    onMonitoring(np.sum(np.subtract(output, y)) / len(y))
                    monitoring_counter = 0

                self.__backprop(output, y)

    def __feedForward(self, x, y):
        output = self.layers[0].feedforward(x)
        for layer in self.layers[1:]:
            output = layer.feedforward(output)
        return output

    def __backprop(self, output, y):
        error = np.multiply(
            self.costFunction.delta(self.layers[-1]._current_weighted_input, output, y),
            self.last_layer_activation_function.derivative(self.layers[-1]._current_weighted_input)
        )

        for layer in reversed(self.layers):
            error = layer.backpropagate(error, self.learningRate, self.mini_batch_size, self.gradient_decent)