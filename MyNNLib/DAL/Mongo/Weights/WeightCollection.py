from DAL.Mongo.CollectionBase import CollectionBase
from DAL.Mongo.Weights.Getter import Getter
from DAL.Mongo.Weights.Saver import Saver


class WeightCollection(CollectionBase):
    def __init__(self, db):
        super().__init__(db, "Weight")
        self.getter = Getter(self.me)
        self.saver = Saver(self.me)

    def getWeights(self, layerId):
        return self.getter.getWeights(layerId)

    def saveWeights(self, w, layerId):
        self.saver.saveWeights(w, layerId)