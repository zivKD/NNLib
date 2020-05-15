from DAL.Mongo.CollectionExecuter import CollectionExectuer

class Getter(CollectionExectuer):
    def __init__(self, collection):
        super().__init__(collection)

    def getLocalReceptiveFieldSize(self, layerId):
        json = self.getScalarByLayerIdAndType(layerId, "local_receptive_field_size")
        return [json.H, json.W]