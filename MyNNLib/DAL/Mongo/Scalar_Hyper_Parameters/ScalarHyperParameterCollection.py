from DAL.Mongo.CollectionBase import CollectionBase
from DAL.Mongo.Scalar_Hyper_Parameters.Getter import Getter
from DAL.Mongo.Scalar_Hyper_Parameters.Saver import Saver


class ScalarHyperParameterCollection(CollectionBase):
    def __init__(self, db):
        super().__init__(db, "Scalar_Hyper_Parameter")
        self.getter = Getter(self.me)
        self.saver = Saver(self.me)

    def getLearningRate(self):
        return self.getter.getLearningRate()

    def getRegularizationTerm(self):
        return self.getter.getRegularizationTerm()

    def getSizeOfMiniBatch(self):
        return self.getter.getSizeOfMiniBatch()

    def getNumberOfEpoches(self):
        return self.getter.getNumberOfEpoches()

    def getStride(self, layerId):
        return self.getStride(layerId)

    def saveLearingRate(self, n):
        self.saver.saveLearingRate(n)

    def saveRegularizationTerm(self, lambada):
        self.saver.saveRegularizationTerm(lambada)

    def saveSizeOfMiniBatch(self, size):
        self.saver.saveSizeOfMiniBatch(size)

    def saveNumberOfEpoches(self, num):
        self.saver.saveNumberOfEpoches(num)

    def saveStride(self, stride, layerId):
        self.saver.saveStride(stride, layerId)