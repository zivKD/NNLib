from abc import ABC, abstractmethod

class BaseDB(ABC):
    @abstractmethod
    def saveWeights(self, w, layerId):
        pass

    @abstractmethod
    def saveBiases(self, b, layerId):
        pass

    @abstractmethod
    def saveLearningRate(self, n):
        pass

    @abstractmethod
    def saveSizeOfMiniBatch(self, m):
        pass

    @abstractmethod
    def saveRegularizationTerm(self, lambada):
        pass

    @abstractmethod
    def saveNumberOfEpoches(self, num):
        pass

    @abstractmethod
    def saveLocalReceptiveSize(self, size, layerId):
        pass

    @abstractmethod
    def saveStride(self, stride, layerId):
        pass

    @abstractmethod
    def getWeights(self, layerId):
        pass

    @abstractmethod
    def getBiases(self, layerId):
        pass

    @abstractmethod
    def getLearningRate(self):
        pass

    @abstractmethod
    def getSizeOfMiniBatch(self):
        pass

    @abstractmethod
    def getRegularizationTerm(self):
        pass

    @abstractmethod
    def getNumberOfEpoches(self):
        pass

    @abstractmethod
    def getLocalReceptiveSize(self, layerId):
        pass

    @abstractmethod
    def getStride(self, layerId):
        pass