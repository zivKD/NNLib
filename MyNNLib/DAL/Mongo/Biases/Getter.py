from DAL.Mongo.CollectionExecuter import CollectionExectuer

class Getter(CollectionExectuer):
    def __init__(self, collection):
        super().__init__(collection)

    def getBiases(self, layerId):
        return self.getColByLayerIdAndType(layerId, "bias")