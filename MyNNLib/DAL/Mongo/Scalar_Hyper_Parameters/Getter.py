from DAL.Mongo.CollectionExecuter import _CollectionExectuer

class _Getter(_CollectionExectuer):
    def __init__(self, collection):
        super().__init__(collection)

    def getLearningRate(self, networkId):
        return self.getScalarByType("learning_rate", networkId)

    def getRegularizationTerm(self, networkId):
        return self.getScalarByType("regularization_term", networkId)

    def getSizeOfMiniBatch(self, networkId):
        return self.getScalarByType("size_of_mini_batch", networkId)

    def getNumberOfEpoches(self, networkId):
        return self.getScalarByType("number_of_epoches", networkId)

    def getStride(self, layerId, networkId):
        return self.getScalarByLayerIdAndType(layerId, "stride", networkId)