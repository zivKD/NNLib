from BL.Activation_Functions.RELU import RELU
from BL.Activation_Functions.Softmax import Softmax
from BL.Cost_Functions.Log_likelihood import Log_likelihood
from BL.DataLoader import DataLoader
from BL.HyperParameterContainer import HyperParameterContainer
from BL.Layers.Convolutional import Convolutional
from BL.Layers.FullyConnected import FullyConnected
from BL.Layers.MaxPooling import MaxPooling
from BL.Network import Network
from BL.Gradient_Decent.MomentumBased import MomentumBased
from BL.Regularization.Dropout import Dropout
from DAL.Mongo.MongoDB import MongoDB

# TODO: Design the system better so it will keep all the SOLID principles and functionality programming
# TODO: Find a replacement to a loop on the mini-batch
# TODO: The reason there is the NaN problem because the values after the softmax activation are too small \
#  so python converts them to Nan. Another problem is that the input are affected greatly, because in the first round \
#  the values are not so small, but for some reason the inputs become very big\small even though in first look the weights\biases \
#  look normal

import numpy as np

training_set, validation_set, test_set = DataLoader.load()
activationFunc = RELU()
HyperParameterContainer.init(
    gradientDescent = MomentumBased(7)
)
dropout = Dropout((5, 6, 7), number_of_runs=10)

layers = [
    Convolutional(
        sizeOfLocalReceptiveField=(5, 5),
        numberOfFilters=20,
        input_image_dims=(28, 28),
        stride=1,
        numberOfInputFeatureMaps=1
    ),
    MaxPooling(
        input_image_dims=(24, 24),
        stride=1,
        poolSize=(2, 2),
        number_of_input_feature_maps=20
    ),
    Convolutional(
        sizeOfLocalReceptiveField=(5, 5),
        numberOfFilters=40,
        input_image_dims=(12, 12),
        stride=1,
        numberOfInputFeatureMaps=20
    ),
    MaxPooling(
        input_image_dims=(12, 12),
        stride=1,
        poolSize=(2, 2),
        number_of_input_feature_maps=40
    ),
    FullyConnected(
        n_in=40 * 4 * 4,
        n_out=1000
    ),
    FullyConnected(
        n_in=1000,
        n_out=1000
    ),
    FullyConnected(
        n_in=1000,
        n_out=10,
        isSoftmax=True
    )
]

model = Network(
    costFunction=Log_likelihood(),
    last_layer_activation_function=Softmax(),
    layers=layers,
    training_set=training_set,
    validation_set=validation_set,
    test_set=test_set,
    db=MongoDB(),
    should_load_from_db=False,
    should_save_to_db=True,
    network_id="1",
    should_regulate=True,
    regularizationTechs=(dropout,)
)

model.runNetwork()
