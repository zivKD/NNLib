from BL.Activation_Functions.RELU import RELU
from BL.BaseClasses.ActivationFunction import ActivationFunction
from BL.BaseClasses.GradientDescent import GradientDescent
from BL.Gradient_Decent.Stochastic import Stochastic


class HyperParameterContainer():
    learningRate : float = None
    regularizationParameter : float = None
    mini_batch_size : int = None
    number_of_epochs : int = None
    dropoutPrecentage : float = None
    activationFunction : ActivationFunction = None
    gradientDescent : GradientDescent = None

    @staticmethod
    def init(
            learningRate : float = 0.03,
            regularizationParameter : float = 1 ,
            mini_batch_size : int = 10,
            number_of_epochs : int = 40,
            dropoutPrecentage : float = 0.5,
            activationFunction : ActivationFunction = RELU(),
            gradientDescent : GradientDescent = Stochastic()
    ):
        HyperParameterContainer.learningRate = learningRate
        HyperParameterContainer.regularizationParameter = regularizationParameter
        HyperParameterContainer.mini_batch_size = mini_batch_size
        HyperParameterContainer.number_of_epochs = number_of_epochs
        HyperParameterContainer.dropoutPrecentage = dropoutPrecentage
        HyperParameterContainer.activationFunction = activationFunction
        HyperParameterContainer.gradientDescent = gradientDescent



