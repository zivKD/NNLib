from DAL.Mongo.CollectionExecuter import CollectionExectuer

class Saver(CollectionExectuer):
    def __init__(self, collection):
        super().__init__(collection)

    def saveWeights(self, w, layerId):
        self.saveColByLayerIdAndType(w, layerId, "weight")