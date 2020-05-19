from DAL.Mongo.CollectionBase import _CollectionBase
from DAL.Mongo.Weights.Getter import _Getter
from DAL.Mongo.Weights.Saver import _Saver


class _WeightCollection(_CollectionBase):
    def __init__(self, db):
        super().__init__(db, "Weight")
        self.getter = _Getter(self.me)
        self.saver = _Saver(self.me)

    def getWeights(self, layerId):
        return self.getter.getWeights(layerId)

    def saveWeights(self, w, layerId):
        self.saver.saveWeights(w, layerId)