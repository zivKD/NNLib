from BL.Activation_Functions.RELU import RELU
from BL.BaseClasses.ActivationFunction import ActivationFunction

class HyperParameterContainer():
    learningRate : float = None
    regularizationParameter : float = None
    mini_batch_size : int = None
    number_of_epochs : int = None
    dropoutPercentage : float = None
    activationFunction : ActivationFunction = None
    gradientDescent = None

    @staticmethod
    def init(
            learningRate : float = 0.03,
            regularizationParameter : float = 1 ,
            gradientDescent = None,
            mini_batch_size : int = 10,
            number_of_epochs : int = 40,
            dropoutPrecentage : float = 0.5,
            activationFunction : ActivationFunction = RELU(),
    ):
        HyperParameterContainer.learningRate = learningRate
        HyperParameterContainer.regularizationParameter = regularizationParameter
        HyperParameterContainer.mini_batch_size = mini_batch_size
        HyperParameterContainer.number_of_epochs = number_of_epochs
        HyperParameterContainer.dropoutPercentage = dropoutPrecentage
        HyperParameterContainer.activationFunction = activationFunction
        HyperParameterContainer.gradientDescent = gradientDescent



