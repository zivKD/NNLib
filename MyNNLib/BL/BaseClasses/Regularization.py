from abc import ABC, abstractmethod

class Regularization(ABC):
    def __init__(self, number_of_layers_to_affect : tuple):
        self._number_of_layers_to_affect = number_of_layers_to_affect

    @abstractmethod
    def changeParams(self, w, b, layerNumber):
        pass