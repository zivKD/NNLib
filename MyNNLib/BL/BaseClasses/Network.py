from BL.Activation_Functions.Sigmoid import Sigmoid
from BL.Cost_Functions.Quadratic import Quadratic
from BL.Layers.Convolutional import Convolutional
from DAL.Mongo.MongoDB import MongoDB


class Network ():
    def __init__(self,
                 costFunction = Quadratic(Sigmoid()),
                 layers = (Convolutional(),),
                 training_set = (),
                 validation_set = (),
                 test_set = (),
                 db = MongoDB()):
        self.costFunction = costFunction
        self.layers = layers
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.db = db


    #TODO: seperate to mini-batches
    def runNetwork(self, frequencyOfPrintingAccuracy):
        # for layer in self.layers:
        #     layer.feedforward(self.training_set)
        pass

