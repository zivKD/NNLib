from DAL.Mongo.Biases.Getter import _Getter
from DAL.Mongo.Biases.Saver import _Saver
from DAL.Mongo.CollectionBase import _CollectionBase

class _BiasCollection(_CollectionBase):
    def __init__(self, db):
        super().__init__(db, "Bias")
        self.getter = _Getter(self.me)
        self.saver = _Saver(self.me)

    def getBiases(self, layerId):
        return self.getter.getBiases(layerId)

    def saveBiases(self, b, layerId):
        self.saver.saveBiases(b, layerId)