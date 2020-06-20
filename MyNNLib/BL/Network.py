from BL.BaseClasses.ActivationFunction import ActivationFunction
from BL.BaseClasses.CostFunction import CostFunction
from BL.BaseClasses.Layer import Layer
from BL.BaseClasses.Regularization import Regularization
from BL.Gradient_Decent.MomentumBased import MomentumBased
from BL.HyperParameterContainer import HyperParameterContainer
from DAL.BaseDB import BaseDB
import numpy as np

class Network():
    #region Init
    def __init__(self,
                 costFunction : CostFunction,
                 last_layer_activation_function : ActivationFunction,
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
        self.learningRate = HyperParameterContainer.learningRate
        self.last_layer_activation_function = last_layer_activation_function
        self.gradient_decent = HyperParameterContainer.gradientDescent
        self.mini_batch_size = HyperParameterContainer.mini_batch_size
        self.number_of_epochs = HyperParameterContainer.mini_batch_size
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
    #endregion

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
            monitoring_counter = 1
            for mini_batch in mini_batches:
                x = np.array([batch[0].ravel() for batch in mini_batch]).transpose()
                y = np.array([batch[1].ravel() for batch in mini_batch]).transpose()
                output = self.__feedForward(x)
                if (monitoring_counter == frequencyOfMonitoring + 1):
                    accuracy = np.abs(np.sum(np.subtract(output, y))) * HyperParameterContainer.mini_batch_size
                    onMonitoring(accuracy)
                    monitoring_counter = 1
                self.__backprop(output, y)
                print("epoch:" + str(i+1) +  " batch:" + str(monitoring_counter))
                monitoring_counter += 1


        if(self.should_save_to_db):
            self.__saveToDb()


    def __feedForward(self, x):
        if(self.__should_regulate):
            for regularization in self.__regularizationTechs:
                for layer in self.layers:
                    layer.regulate(regularization)
        output = x
        for layer in self.layers:
            output = layer.feedforward(output)
        return output


    def __backprop(self, output, y):
        error = np.multiply(
            self.costFunction.derivative(self.layers[-1]._current_weighted_input, output, y),
            self.last_layer_activation_function.derivative(self.layers[-1]._current_weighted_input)
        )
        for layer in reversed(self.layers):
            error = layer.backpropagate(error)

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