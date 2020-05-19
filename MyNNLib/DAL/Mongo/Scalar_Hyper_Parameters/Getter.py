from DAL.Mongo.CollectionExecuter import _CollectionExectuer

class _Getter(_CollectionExectuer):
    def __init__(self, collection):
        super().__init__(collection)

    def getLearningRate(self):
        return self.getScalarByType("learning_rate")

    def getRegularizationTerm(self):
        return self.getScalarByType("regularization_term")

    def getSizeOfMiniBatch(self):
        return self.getScalarByType("size_of_mini_batch")

    def getNumberOfEpoches(self):
        return self.getScalarByType("number_of_epoches")

    def getStride(self, layerId):
        return self.getScalarByLayerIdAndType(layerId, "stride")