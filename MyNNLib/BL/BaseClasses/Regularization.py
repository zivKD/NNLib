from abc import ABC, abstractmethod

class Regularization(ABC):
    @abstractmethod
    def changeParams(self, w, b, layerNumber):
        pass