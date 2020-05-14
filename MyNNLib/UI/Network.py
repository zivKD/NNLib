class Network ():
    def __init__(self, costFunction, layers, training_set, validation_set, test_set):
        self.costFunction = costFunction
        self.layers = layers
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set

    def runNetwork(self, frequencyOfPrintingAccuracy, time):
        while(1 == 1):
            output = self.layers[0].feedForward(self.training_set)
            for layer in self.layers[1:]:
               output = layer.feedforward(output)

            error = self.costFunction.derivativeByActivation(output) * self.costFunction.derivativeByWeightedInput()
