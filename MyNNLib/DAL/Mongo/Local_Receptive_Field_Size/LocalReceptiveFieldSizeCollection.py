from DAL.Mongo.CollectionBase import _CollectionBase
from DAL.Mongo.Local_Receptive_Field_Size.Getter import _Getter
from DAL.Mongo.Local_Receptive_Field_Size.Saver import _Saver


class _LocalReceptiveFieldSizeCollection(_CollectionBase):
    def __init__(self, db):
        super().__init__(db, "Weight")
        self.getter = _Getter(self.me)
        self.saver = _Saver(self.me)

    def saveLocalReceptiveFieldSize(self, size, layerId):
        self.saver.saveLocalReceptiveSize(size, layerId)

    def getLocalReceptiveField(self, layerId):
        self.getter.getLocalReceptiveFieldSize(layerId)

