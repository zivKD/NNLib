from DAL.Mongo.CollectionExecuter import _CollectionExectuer

class _Saver(_CollectionExectuer):
    def __init__(self, collection):
        super().__init__(collection)

    def saveWeights(self, w, layerId, networkId):
        self.saveColByLayerIdAndType(w, layerId, "weight", networkId)