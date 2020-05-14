
class InvalidInputException(Exception):
    def __init__(self, number):
        super().__init__("The input to the " + number + "th layer is invalid")