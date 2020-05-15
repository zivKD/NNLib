from DAL.Mongo.Biases.Getter import Getter
from DAL.Mongo.Biases.Saver import Saver
from DAL.Mongo.CollectionBase import CollectionBase

class BiasCollection(CollectionBase):
    def __init__(self, db):
        super().__init__(db, "Bias")
        self.getter = Getter(self.me)
        self.saver = Saver(self.me)

    def getBiases(self, layerId):
        return self.getter.getBiases(layerId)

    def saveBiases(self, b, layerId):
        self.saver.saveBiases(b, layerId)