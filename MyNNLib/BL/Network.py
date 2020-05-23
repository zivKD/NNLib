from random import random
from BL.BaseClasses.ActivationFunction import ActivationFunction
from BL.BaseClasses.CostFunction import CostFunction
from BL.BaseClasses.GradientDescent import GradientDescent
from BL.BaseClasses.Layer import Layer
from BL.BaseClasses.Regularization import Regularization
from BL.Gradient_Decent.MomentumBased import MomentumBased
from DAL.BaseDB import BaseDB
import numpy as np

class Network():
    def __init__(self,
                 costFunction : CostFunction,
                 learningRate : float,
                 last_layer_activation_function : ActivationFunction,
                 gradient_descent : GradientDescent,
                 mini_batch_size : int,
                 number_of_epoches : int,
                 layers : [Layer,],
                 db: BaseDB,
                 should_load_from_db: bool,
                 should_save_to_db : bool,
                 network_id : str,
                 should_regulate : bool,
                 training_set=(),
                 validation_set=(),
                 test_set=(),
                 regularizationTechs : (Regularization,) = None
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
        self.__should_regulate = should_regulate
        self.__regularizationTechs = regularizationTechs

    def runNetwork(self, onMonitoring=lambda accuracy: print(accuracy), frequencyOfMonitoring=10):
        if(self.should_load_from_db):
            self.__getFromDb()

        for i in range(self.number_of_epochs):
            n = len(self.training_set)
            np.random.shuffle(self.training_set)
            mini_batches = [
                self.training_set[k:k + self.mini_batch_size]
                for k in range(0, n, self.mini_batch_size)
            ]

            monitoring_counter = 0
            for mini_batch in mini_batches:
                monitoring_counter += 1
                x = np.array([batch[0].ravel() for batch in mini_batch]).transpose()
                y = np.array([batch[1].ravel() for batch in mini_batch]).transpose()
                output = self.__feedForward(x)
                if (monitoring_counter == frequencyOfMonitoring):
                    onMonitoring(np.sum(np.subtract(output, y)) / len(y))
                    monitoring_counter = 0

                self.__backprop(output, y)

        if(self.should_save_to_db):
            self.__saveToDb()

    def __feedForward(self, x):
        output = None
        if((self.gradient_decent) is MomentumBased):
            counter = 0
            output = self.layers[0].feedforward(x)
            self.gradient_decent.setVelocityMatrix(counter, self.layers[counter].getWeightShape(),
                                                   self.layers[counter].getBiasShape())
            for layer in self.layers[1:]:
                counter+=1
                output = layer.feedforward(output)
                self.gradient_decent.setVelocityMatrix(counter, self.layers[counter].getWeightShape(),
                                                       self.layers[counter].getBiasShape())
        else:
            output = self.layers[0].feedforward(x)
            for layer in self.layers[1:]:
                output = layer.feedforward(output)

        return output


    def __backprop(self, output, y):
        error = np.multiply(
            self.costFunction.derivative(self.layers[-1]._current_weighted_input, output, y),
            self.last_layer_activation_function.derivative(self.layers[-1]._current_weighted_input)
        )

        if(self.__should_regulate):
            for regularization in self.__regularizationTechs:
                for layer in self.layers:
                    layer.regulate(regularization)

        for layer in reversed(self.layers):
            error = layer.backpropagate(error, self.learningRate, self.mini_batch_size, self.gradient_decent)

    def __saveToDb(self):
        for layer in self.layers:
            layer.saveToDb(self.db, self.id)
        self.db.saveLearningRate(self.learningRate, self.id)
        self.db.saveNumberOfEpoches(self.number_of_epochs, self.id)
        self.db.saveSizeOfMiniBatch(self.mini_batch_size, self.id)

    def __getFromDb(self):
        for layer in self.layers:
            layer.getFromDb(self.db, self.id)
        self.learningRate = self.db.getLearningRate(self.id)
        self.number_of_epochs = self.db.getNumberOfEpoches(self.id)
        self.mini_batch_size = self.db.getSizeOfMiniBatch(self.id)