from DAL.Mongo.CollectionExecuter import _CollectionExectuer

class _Saver(_CollectionExectuer):
    def __init__(self, collection):
        super().__init__(collection)

    def saveLearingRate(self, n, networkId):
        self.saveScalarByType(n, "learning_rate", networkId)

    def saveRegularizationTerm(self, lambada, networkId):
        self.saveScalarByType(lambada, "regularization_term", networkId)

    def saveSizeOfMiniBatch(self, size, networkId):
        self.saveScalarByType(size, "size_of_mini_batch", networkId)

    def saveNumberOfEpoches(self, num, networkId):
        self.saveScalarByType(num, "number_of_epoches", networkId)

    def saveStride(self, stride, layerId, networkId):
        self.saveScalarByLayerIdAndType(stride, layerId, "stride", networkId)


