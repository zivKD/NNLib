from abc import ABC, abstractmethod

class BaseDB(ABC):
    @abstractmethod
    def saveWeights(self, w, layerId, networkId):
        pass

    @abstractmethod
    def saveBiases(self, b, layerId, networkId):
        pass

    @abstractmethod
    def saveLearningRate(self, n, networkId):
        pass

    @abstractmethod
    def saveSizeOfMiniBatch(self, m, networkId):
        pass

    @abstractmethod
    def saveRegularizationTerm(self, lambada, networkId):
        pass

    @abstractmethod
    def saveNumberOfEpoches(self, num, networkId):
        pass

    @abstractmethod
    def saveLocalReceptiveSize(self, size, layerId, networkId):
        pass

    @abstractmethod
    def saveStride(self, stride, layerId, networkId):
        pass

    @abstractmethod
    def getWeights(self, layerId, networkId):
        pass

    @abstractmethod
    def getBiases(self, layerId, networkId):
        pass

    @abstractmethod
    def getLearningRate(self, networkId):
        pass

    @abstractmethod
    def getSizeOfMiniBatch(self, networkId):
        pass

    @abstractmethod
    def getRegularizationTerm(self, networkId):
        pass

    @abstractmethod
    def getNumberOfEpoches(self, networkId):
        pass

    @abstractmethod
    def getLocalReceptiveSize(self, layerId, networkId):
        pass

    @abstractmethod
    def getStride(self, layerId, networkId):
        pass