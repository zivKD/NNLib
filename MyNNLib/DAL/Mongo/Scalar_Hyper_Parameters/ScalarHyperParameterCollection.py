from DAL.Mongo.CollectionBase import _CollectionBase
from DAL.Mongo.Scalar_Hyper_Parameters.Getter import _Getter
from DAL.Mongo.Scalar_Hyper_Parameters.Saver import _Saver


class _ScalarHyperParameterCollection(_CollectionBase):
    def __init__(self, db):
        super().__init__(db, "Scalar_Hyper_Parameter")
        self.getter = _Getter(self.me)
        self.saver = _Saver(self.me)

    def getLearningRate(self, networkId):
        return self.getter.getLearningRate(networkId)

    def getRegularizationTerm(self, networkId):
        return self.getter.getRegularizationTerm(networkId)

    def getSizeOfMiniBatch(self, networkId):
        return self.getter.getSizeOfMiniBatch(networkId)

    def getNumberOfEpoches(self, networkId):
        return self.getter.getNumberOfEpoches(networkId)

    def getStride(self, layerId, networkId):
        return self.getStride(layerId, networkId)

    def saveLearingRate(self, n, networkId):
        self.saver.saveLearingRate(n, networkId)

    def saveRegularizationTerm(self, lambada, networkId):
        self.saver.saveRegularizationTerm(lambada, networkId)

    def saveSizeOfMiniBatch(self, size, networkId):
        self.saver.saveSizeOfMiniBatch(size, networkId)

    def saveNumberOfEpoches(self, num, networkId):
        self.saver.saveNumberOfEpoches(num, networkId)

    def saveStride(self, stride, layerId, networkId):
        self.saver.saveStride(stride, layerId, networkId)