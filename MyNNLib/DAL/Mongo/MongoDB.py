from DAL.BaseDB import BaseDB
import pymongo
import Config as conf
from DAL.Mongo.Biases.BiasCollection import _BiasCollection
from DAL.Mongo.Local_Receptive_Field_Size.LocalReceptiveFieldSizeCollection import _LocalReceptiveFieldSizeCollection
from DAL.Mongo.Scalar_Hyper_Parameters.ScalarHyperParameterCollection import _ScalarHyperParameterCollection
from DAL.Mongo.Weights.WeightCollection import _WeightCollection


class MongoDB(BaseDB):
    def _init(self, dbName):
        myclient = pymongo.MongoClient(conf.mongo["url"])
        mydb = myclient[dbName]
        self.biasCol = _BiasCollection(mydb)
        self.weightCol = _WeightCollection(mydb)
        self.lrfsCol = _LocalReceptiveFieldSizeCollection(mydb)
        self.scalarCol = _ScalarHyperParameterCollection(mydb)
        

    def saveWeights(self, w, layerId, networkId):
        self.weightCol.saveWeights(w, layerId, networkId)

    def saveBiases(self, b, layerId, networkId):
        self.biasCol.saveBiases(b, layerId, networkId)

    def saveLearningRate(self, n, networkId):
        self.scalarCol.saveLearingRate(n, networkId)

    def saveSizeOfMiniBatch(self, m, networkId):
        self.scalarCol.saveSizeOfMiniBatch(m, networkId)

    def saveRegularizationTerm(self, lambada, networkId):
        self.scalarCol.saveRegularizationTerm(lambada, networkId)

    def saveNumberOfEpoches(self, num, networkId):
        self.scalarCol.saveNumberOfEpoches(num, networkId)

    def saveLocalReceptiveSize(self, size, layerId, networkId):
        self.lrfsCol.saveLocalReceptiveFieldSize(size, layerId, networkId)

    def saveStride(self, stride, layerId, networkId):
        self.scalarCol.saveStride(stride, layerId, networkId)

    def getWeights(self, layerId, networkId):
        self.weightCol.getWeights(layerId, networkId)

    def getBiases(self, layerId, networkId):
        self.biasCol.getBiases(layerId, networkId)

    def getLearningRate(self, networkId):
        self.scalarCol.getLearningRate(networkId)

    def getSizeOfMiniBatch(self, networkId):
        self.scalarCol.getSizeOfMiniBatch(networkId)

    def getRegularizationTerm(self, networkId):
        self.scalarCol.getRegularizationTerm(networkId)

    def getNumberOfEpoches(self, networkId):
        self.scalarCol.getNumberOfEpoches(networkId)

    def getLocalReceptiveSize(self, layerId, networkId):
        self.lrfsCol.getLocalReceptiveField(layerId, networkId)

    def getStride(self, layerId, networkId):
        self.scalarCol.getStride(layerId, networkId)


