
class _CollectionExectuer():
    def __init__(self, collection):
        self.collection = collection

    def getColByLayerIdAndType(self, layerId, type):
        col = self.collection.find({"layerId": layerId, "type": type})
        return [json.value for json in col]

    def getScalarByLayerIdAndType(self, layerId, type):
        return self.collection.find({"layerId": layerId, "type": type}).value

    def getScalarByType(self, type):
        return self.collection.find({"type": type}).value

    def saveColByLayerIdAndType(self, col, layerId, type):
        list = [
            { "value" : item, "layerId": layerId, "type": type}
            for item in col
        ]

        self.collection.insert_many(list)

    def saveScalarByType(self, scalar, type):
        self.collection.insert_one({"type": type, "value": scalar})

    def saveScalarByLayerIdAndType(self, scalar, layerId, type):
        self.collection.insert_one({"type": type, "value": scalar, "layerId": layerId})
