import numpy as np
import warnings

class _MathHelper():
    @staticmethod
    def IsNan(matrix):
        return np.isnan(np.sum(matrix))

    @staticmethod
    def repeat(matrix, axis, num_of_repeats, should_expand=(True,)):
        if(type(matrix) != 'numpy.ndarray'):
            matrix = np.array(matrix)
        if(type(axis) == int):
            if((type(should_expand) == tuple and should_expand[0]) or should_expand):
                matrix = np.expand_dims(matrix, axis=axis)
            matrix = matrix.repeat(num_of_repeats, axis=axis)
        else:
            if(len(should_expand) == len(axis)):
                for i in range(len(axis)):
                    if(should_expand[i]):
                        matrix = np.expand_dims(matrix, axis=axis[i])
            for i in range(len(axis)):
                matrix = matrix.repeat(num_of_repeats[i], axis=axis[i])

        return matrix

    @staticmethod
    def get_output_image_dims(imageDims, lrfDims, stride):
        return (
            (1 + (imageDims[0] - lrfDims[0]) // stride),
            (1 + (imageDims[1] - lrfDims[1]) // stride)
        )

    @staticmethod
    def get_num_of_local_receptive_fields(imageDims, lrfDims, stride,
                                          numberOfInputFeatureMaps, numberOfFilters):
        return _MathHelper.get_output_image_dims(imageDims, lrfDims, stride) * numberOfFilters * numberOfInputFeatureMaps

    @staticmethod
    def get_local_receptive_fields(matrix, stride, outputImageDims, lrfDims):
        s0, s1 = matrix.strides[-2:]
        matrix_dims = np.ndim(matrix)
        view_shape = matrix.shape[:2-matrix_dims] + (outputImageDims[0], outputImageDims[1], lrfDims[0], lrfDims[1])
        strides = matrix.strides[:2-matrix_dims] + (stride * s0, stride * s1, s0, s1)
        return np.lib.stride_tricks.as_strided(matrix, view_shape, strides=strides)

    @staticmethod
    def pad_arr(var, pad):
        p = ([0, 0],) * (np.ndim(var) - 2) + ([pad, pad], [pad, pad])
        var_pad = np.pad(var, p, mode='constant', constant_values=0)
        return var_pad

    @staticmethod
    def conv_with_warnings(matrix, kernel, stride=1, pad=0):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                _MathHelper.conv(matrix, kernel, stride, pad)
            except Warning as warn:
                print(warn)

    @staticmethod
    def conv(matrix, kernel, stride=1, pad=0):
        if pad > 0:
            matrix = _MathHelper.pad_arr(matrix, pad)
        x, y = _MathHelper.get_output_image_dims(matrix.shape[-2:], kernel.shape[-2:], stride)
        kernel = np.repeat(kernel[:, :, None, :, :], x, axis=2)
        kernel = np.repeat(kernel[:, :, :, None, :, :], y, axis=3)
        recepted_matrix = _MathHelper.get_local_receptive_fields(matrix, stride, (x, y), kernel.shape[-2:])
        product = np.multiply(recepted_matrix, kernel)
        convolution_product = np.sum(product, axis=(4, 5))
        return convolution_product
