from DAL.Mongo.CollectionExecuter import _CollectionExectuer

class _Saver(_CollectionExectuer):
    def __init__(self, collection):
        super().__init__(collection)

    def saveWeights(self, w, layerId):
        self.saveColByLayerIdAndType(w, layerId, "weight")