from DAL.Mongo.CollectionExecuter import _CollectionExectuer

class _Saver(_CollectionExectuer):
    def __init__(self, collection):
        super().__init__(collection)

    def saveBiases(self, b, layerId, networkId):
        self.saveColByLayerIdAndType(b, layerId, "bias", networkId)