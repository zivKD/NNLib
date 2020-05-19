from random import random
from BL.BaseClasses.ActivationFunction import ActivationFunction
from BL.BaseClasses.CostFunction import CostFunction
from BL.BaseClasses.GradientDescent import GradientDescent
from BL.BaseClasses.Layer import Layer
from BL.Gradient_Decent import MomentumBased
from DAL.BaseDB import BaseDB
import numpy as np


class Network():
    def __init__(self,
                 costFunction : CostFunction,
                 learningRate : int,
                 last_layer_activation_function : ActivationFunction,
                 gradient_descent : GradientDescent,
                 mini_batch_size : int,
                 number_of_epoches : int,
                 layers : [Layer,...],
                 db: BaseDB,
                 should_load_from_db: bool,
                 should_save_to_db : bool,
                 network_id : str,
                 training_set=(),
                 validation_set=(),
                 test_set=(),
                 ):
        self.costFunction = costFunction
        self.learningRate = learningRate
        self.last_layer_activation_function = last_layer_activation_function
        self.gradient_decent = gradient_descent
        self.mini_batch_size = mini_batch_size
        self.number_of_epochs = number_of_epoches
        self.layers = layers
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.db = db
        self.should_load_from_db = should_load_from_db
        self.should_save_to_db = should_save_to_db
        self.id = network_id

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
        if((self.gradient_decent) is MomentumBased):
            counter = 0
            output = self.layers[0].feedforward(x)
            self.gradient_decent.setVelocityMatrix(counter, self.layers[counter])
            for layer in self.layers[1:]:
                counter+=1
                output = layer.feedforward(output)
                self.gradient_decent.setVelocityMatrix(counter, self.layers[counter])
            return output
        else:
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

    def _saveToDb(self):
        for layer in self.layers:
            layer.saveToDb(self.db)
        self.db.saveLearningRate(self.learningRate)
        self.db.saveNumberOfEpoches(self.number_of_epochs)
        self.db.saveSizeOfMiniBatch(self.mini_batch_size)