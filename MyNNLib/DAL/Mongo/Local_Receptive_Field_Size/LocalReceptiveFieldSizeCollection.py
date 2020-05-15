from DAL.Mongo.CollectionBase import CollectionBase
from DAL.Mongo.Local_Receptive_Field_Size.Getter import Getter
from DAL.Mongo.Local_Receptive_Field_Size.Saver import Saver


class LocalReceptiveFieldSizeCollection(CollectionBase):
    def __init__(self, db):
        super().__init__(db, "Weight")
        self.getter = Getter(self.me)
        self.saver = Saver(self.me)

    def saveLocalReceptiveFieldSize(self, size, layerId):
        self.saver.saveLocalReceptiveSize(size, layerId)

    def getLocalReceptiveField(self, layerId):
        self.getter.getLocalReceptiveFieldSize(layerId)

