from DAL.Mongo.CollectionExecuter import _CollectionExectuer

class _Saver(_CollectionExectuer):
    def __init__(self, collection):
        super().__init__(collection)

    def saveLocalReceptiveSize(self, size, layerId):
        self.collection.insert_one({"H" : size[0], "W": size[1], layerId:layerId})