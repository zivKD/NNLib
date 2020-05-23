import gzip

from BL.Activation_Functions.RELU import RELU
from BL.Activation_Functions.Softmax import Softmax
from BL.Cost_Functions.Log_likelihood import Log_likelihood
from BL.DataLoader import DataLoader
from BL.Layers.ConvolutionalLayer.Convolutional import Convolutional
from BL.Layers.FullyConnected import FullyConnected
from BL.Layers.MaxPooling import MaxPooling
from BL.Network import Network
from BL.Gradient_Decent.MomentumBased import MomentumBased
from BL.Regularization.Dropout import Dropout
from DAL.Mongo.MongoDB import MongoDB

training_set,validation_set,test_set = DataLoader.load()
activationFunc = RELU()
dropout = Dropout((5,6,7), number_of_runs=10)
layers = [
    Convolutional(
        activationFunc,
        sizeOfLocalReceptiveField=(5,5),
        numberOfFilters=20,
        sizeOfInputImage=(28, 28),
        stride=1,
        numberOfInputFeatureMaps=1
    ),
    MaxPooling(
        sizeOfInputImage=(28, 28),
        stride=1,
        poolSize=(2,2)
    ),
    Convolutional(
        activationFunc,
        sizeOfLocalReceptiveField=(5, 5),
        numberOfFilters=40,
        sizeOfInputImage=(12, 12),
        stride=1,
        numberOfInputFeatureMaps=20
    ),
    MaxPooling(
        sizeOfInputImage=(12, 12),
        stride=1,
        poolSize=(2,2)
    ),
    FullyConnected(
        n_in= 40 * 4 * 4,
        activationFunction = activationFunc,
        n_out=1000
    ),
    FullyConnected(
        n_in = 1000,
        activationFunction = activationFunc,
        n_out=1000
    ),
    FullyConnected(
        n_in = 1000,
        activationFunction=Softmax(),
        n_out=10
    )
]

model = Network(
    costFunction=Log_likelihood(),
    learningRate=0.03,
    last_layer_activation_function=Softmax(),
    gradient_descent= MomentumBased(5),
    mini_batch_size=10,
    number_of_epoches=40,
    layers = layers,
    training_set=training_set,
    validation_set=validation_set,
    test_set=test_set,
    db = MongoDB(),
    should_load_from_db= False,
    should_save_to_db=True,
    network_id= "1",
    should_regulate=True,
    regularizationTechs= (dropout)
)

model.runNetwork()

