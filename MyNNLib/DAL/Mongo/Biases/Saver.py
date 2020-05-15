from DAL.Mongo.CollectionExecuter import CollectionExectuer

class Saver(CollectionExectuer):
    def __init__(self, collection):
        super().__init__(collection)

    def saveBiases(self, b, layerId):
        self.saveColByLayerIdAndType(b, layerId, "bias")