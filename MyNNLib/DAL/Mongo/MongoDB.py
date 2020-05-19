from DAL.BaseDB import BaseDB
import pymongo
import Config as conf
from DAL.Mongo.Biases.BiasCollection import _BiasCollection
from DAL.Mongo.Local_Receptive_Field_Size.LocalReceptiveFieldSizeCollection import _LocalReceptiveFieldSizeCollection
from DAL.Mongo.Scalar_Hyper_Parameters.ScalarHyperParameterCollection import _ScalarHyperParameterCollection
from DAL.Mongo.Weights.WeightCollection import _WeightCollection


class MongoDB(BaseDB):
    def _init(self, dbName):
        super().__init__()
        myclient = pymongo.MongoClient(conf.mongo["url"])
        mydb = myclient[dbName]
        self.biasCol = _BiasCollection(mydb)
        self.weightCol = _WeightCollection(mydb)
        self.lrfsCol = _LocalReceptiveFieldSizeCollection(mydb)
        self.scalarCol = _ScalarHyperParameterCollection(mydb)
        

    def saveWeights(self, w, layerId):
        self.weightCol.saveWeights(w, layerId)

    def saveBiases(self, b, layerId):
        self.biasCol.saveBiases(b, layerId)

    def saveLearningRate(self, n):
        self.scalarCol.saveLearingRate(n)

    def saveSizeOfMiniBatch(self, m):
        self.scalarCol.saveSizeOfMiniBatch(m)

    def saveRegularizationTerm(self, lambada):
        self.scalarCol.saveRegularizationTerm(lambada)

    def saveNumberOfEpoches(self, num):
        self.scalarCol.saveNumberOfEpoches(num)

    def saveLocalReceptiveSize(self, size, layerId):
        self.lrfsCol.saveLocalReceptiveFieldSize(size, layerId)

    def saveStride(self, stride, layerId):
        self.scalarCol.saveStride(stride, layerId)

    def getWeights(self, layerId):
        self.weightCol.getWeights(layerId)

    def getBiases(self, layerId):
        self.biasCol.getBiases(layerId)

    def getLearningRate(self):
        self.scalarCol.getLearningRate()

    def getSizeOfMiniBatch(self):
        self.scalarCol.getSizeOfMiniBatch()

    def getRegularizationTerm(self):
        self.scalarCol.getRegularizationTerm()

    def getNumberOfEpoches(self):
        self.scalarCol.getNumberOfEpoches()

    def getLocalReceptiveSize(self, layerId):
        self.lrfsCol.getLocalReceptiveField(layerId)

    def getStride(self, layerId):
        self.scalarCol.getStride(layerId)


