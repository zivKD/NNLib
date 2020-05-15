from abc import ABC, abstractmethod

class BaseDB(ABC):
    @abstractmethod
    def insertWeights(self, w, layerId):
        pass

    @abstractmethod
    def insertBiases(self, b, layerId):
        pass

    @abstractmethod
    def insertLearningRate(self, n, layerId):
        pass

    @abstractmethod
    def insertSizeOfMiniBatches(self, m, layerId):
        pass

    @abstractmethod
    def insertRegularizationTerm(self, lambada, layerId):
        pass

    @abstractmethod
    def insertNumberOfEpoches(self, num, layerId):
        pass

    @abstractmethod
    def insertLocalReceptiveSize(self, size, layerId):
        pass

    @abstractmethod
    def insertStride(self, stride, layerId):
        pass

    @abstractmethod
    def getByLayerId(self, layerId):
        pass




