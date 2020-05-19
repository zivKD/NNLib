from DAL.Mongo.CollectionExecuter import _CollectionExectuer

class _Saver(_CollectionExectuer):
    def __init__(self, collection):
        super().__init__(collection)

    def saveLearingRate(self, n):
        self.saveScalarByType(n, "learning_rate")

    def saveRegularizationTerm(self, lambada):
        self.saveScalarByType(lambada, "regularization_term")

    def saveSizeOfMiniBatch(self, size):
        self.saveScalarByType(size, "size_of_mini_batch")

    def saveNumberOfEpoches(self, num):
        self.saveScalarByType(num, "number_of_epoches")

    def saveStride(self, stride, layerId):
        self.saveScalarByLayerIdAndType(stride, layerId, "stride")


