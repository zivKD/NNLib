
class _CollectionExectuer():
    def __init__(self, collection):
        self.collection = collection

    def getColByLayerIdAndType(self, layerId, type, networkId):
        col = self.collection.find({"layerId": layerId, "type": type, "networkId" : networkId})
        return [json.value for json in col]

    def getScalarByLayerIdAndType(self, layerId, type, networkId):
        return self.collection.find({"layerId": layerId, "type": type, "networkId" : networkId}).value

    def getScalarByType(self, type, networkId):
        return self.collection.find({"type": type, "networkId" : networkId}).value

    def saveColByLayerIdAndType(self, col, layerId, type, networkId):
        list = [
            { "value" : item, "layerId": layerId, "type": type, "networkId" : networkId}
            for item in col
        ]

        self.collection.insert_many(list)

    def saveScalarByType(self, scalar, type, networkId):
        self.collection.insert_one({"type": type, "value": scalar, "networkId" : networkId})

    def saveScalarByLayerIdAndType(self, scalar, layerId, type, networkId):
        self.collection.insert_one({"type": type, "value": scalar, "layerId": layerId, "networkId" : networkId})
