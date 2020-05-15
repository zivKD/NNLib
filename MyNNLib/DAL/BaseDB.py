from abc import ABC, abstractmethod

class BaseDB(ABC):

    @abstractmethod
    def insertWeights(self, w):
        pass


    @abstractmethod
    def insertBiases(self, b):
        pass

    @abstractmethod
    def insertLearningRate(self, n):
        pass

    @abstractmethod
    def insertSizeOfMiniBatches(self, m):
        pass

    @abstractmethod
    def insertRegularizationTerm(self, lambada):
        pass

    @abstractmethod
    def insertNumberOfEpoches(self, n):
        pass
