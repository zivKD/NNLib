from DAL.Mongo.CollectionExecuter import _CollectionExectuer

class _Getter(_CollectionExectuer):
    def __init__(self, collection):
        super().__init__(collection)

    def getWeights(self, layerId):
        return self.getColByLayerIdAndType(layerId, "weight")