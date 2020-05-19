from abc import ABC, abstractmethod

class _CollectionBase(ABC):
    def __init__(self, db, name):
        self.me = db[name]

    @abstractmethod
    def getFunctionality(self):
        pass
