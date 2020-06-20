from BL.BaseClasses.Layer import Layer
import numpy as np
from BL.HyperParameterContainer import HyperParameterContainer
from BL.Layers.MathHelper import _MathHelper
from DAL.BaseDB import BaseDB

class Convolutional(Layer):
    def __init__(self, sizeOfLocalReceptiveField, input_image_dims, numberOfFilters, numberOfInputFeatureMaps=1, stride = 1):
        super().__init__("CONVOLUTIONAL")
        self.__sizeOfLocalReceptiveField = sizeOfLocalReceptiveField
        self.__stride = stride
        self.__numberOfInputFeatureMaps = numberOfInputFeatureMaps
        self.__input_image_dims = input_image_dims
        self.__numberOfFilters = numberOfFilters
        self.__mathHelper = _MathHelper()
        self.__initialize_filters()
        self._expected_input_shape = (HyperParameterContainer.mini_batch_size, self.__numberOfInputFeatureMaps) + \
                                     self.__input_image_dims[:]
        self._expected_output_shape = (HyperParameterContainer.mini_batch_size, self.__numberOfFilters) + \
                                      self.__output_image_dims[:]

    def __initialize_filters(self):
        mini_batch_size = HyperParameterContainer.mini_batch_size
        self.__output_image_dims = _MathHelper.get_output_image_dims(self.__input_image_dims, self.__sizeOfLocalReceptiveField, self.__stride)
        biases = np.random.normal(loc = 0, scale = 1, size = (self.__numberOfFilters,))
        num_of_repeats = (mini_batch_size,) + self.__output_image_dims
        self._biases = _MathHelper.repeat(biases, axis=(0, 2, 3), num_of_repeats=num_of_repeats, should_expand=(True, True, True))
        scale = np.sqrt(1/(self.__numberOfFilters * np.prod(self.__sizeOfLocalReceptiveField[0:])))
        weights = np.random.normal(loc=0, scale=scale, size= (self.__numberOfFilters, self.__sizeOfLocalReceptiveField[0], self.__sizeOfLocalReceptiveField[1]))
        self._weights = _MathHelper.repeat(weights, axis=0, num_of_repeats=mini_batch_size)

    def feedforward(self, inputs):
        inputs = super()._turnToInputShape(inputs)
        inputs = _MathHelper.repeat(inputs, axis=(1,), num_of_repeats=(self.__numberOfFilters,))
        self._current_input = inputs
        convolution_product = _MathHelper.conv(inputs, self._weights, self.__stride)
        self._current_weighted_input = np.add(self._biases, convolution_product)
        self._current_activation = self._activationFunction.function(self._current_weighted_input)
        return self._current_activation

    def backpropagate(self, error):
        error = super()._turnToOutputShape(error)
        self.change_by_gradient(error)
        return self.calculate_this_layer_error(error)

    def calculate_this_layer_error(self, error):
        flipped_weights = self.__rotBy180D(self._weights)
        flipped_weights = _MathHelper.repeat(flipped_weights, axis=(2,), num_of_repeats=(self.__numberOfInputFeatureMaps,))
        padding = error.shape[-1] - flipped_weights.shape[-1] + 2
        convolution_product = _MathHelper.conv(flipped_weights, error, self.__stride, pad=padding * self.__stride)
        this_layer_error = np.multiply(
            convolution_product,
            self._activationFunction.derivative(self._current_weighted_input)
        )
        # Opposite of striding to local receptive fields
        this_layer_error = _MathHelper.repeat(this_layer_error, axis=1, num_of_repeats=self.__numberOfInputFeatureMaps)
        this_layer_error = _MathHelper.get_local_receptive_fields(this_layer_error, self.__stride,
                                                                  self.__input_image_dims,
                                                                  self.__output_image_dims)
        return np.sum(this_layer_error, axis=(4, 5))

    def change_by_gradient(self, error):
        gradient_descent = HyperParameterContainer.gradientDescent
        self._biases = gradient_descent.changeBiases(self._biases, error, self.number)
        gradient = _MathHelper.conv(self._current_input, error, self.__stride)
        self._weights = gradient_descent.changeWeights(
            self._weights,
            gradient,
            self.number
        )

    def saveToDb(self, db : BaseDB, neworkId):
        super().saveToDb(db, neworkId)
        db.saveStride(self.__stride, self.number, neworkId)

    def getFromDb(self, db : BaseDB, networkId):
        super().saveToDb(db, networkId)
        self.__stride = db.getStride(self.number, networkId)

    def __rotBy180D(self, matrix):
        matrix = [[
            list(zip(*matrix[i][j][::-1]))
            for j in range(len(matrix[0]))]
            for i in range(len(matrix))]
        return np.array([[
            list(zip(*matrix[i][j][::-1]))
            for j in range(len(matrix[0]))]
            for i in range(len(matrix))])
