import numpy as np

from BL.HyperParameterContainer import HyperParameterContainer

class _MathHelper():
    @staticmethod
    def getOutputImageDims(imageWidth, imageHeight, lrfWidth, lrfHeight, stride):
        return [
            (1 + (imageWidth - lrfWidth) // stride),
            (1 + (imageHeight - lrfHeight) // stride)
        ]

    @staticmethod
    def getNumberOfLocalReceptiveFields(imageWidth, imageHeight, lrfWidth, lrfHeight, stride,
                                        numberOfInputFeatureMaps, numberOfFilters):
        return _MathHelper.getOutputImageDims(imageWidth, imageHeight, lrfWidth, lrfHeight, stride) * \
            numberOfFilters * numberOfInputFeatureMaps

    @staticmethod
    def getLocalReceptiveFields(
            matrix,
            stride,
            outputImageWidth,
            outputImageHeight,
            lrfWidth,
            lrfHeight
    ):
        s0, s1 = matrix.strides[-2:]
        matrix_dims = np.ndim(matrix)
        view_shape = matrix.shape[:2-matrix_dims] + (outputImageWidth, outputImageHeight, lrfWidth, lrfHeight)
        strides = matrix.strides[:2-matrix_dims] + (stride * s0, stride * s1, s0, s1)
        return np.lib.stride_tricks.as_strided(matrix, view_shape, strides=strides)

    @staticmethod
    def conv5D(matrix, kernel, stride=1):
        # needed variables
        imageWidth, imageHeight = matrix.shape[-2:]
        localReceptiveFieldWidth, localReceptiveFieldHeight = kernel.shape[-2:]
        x, y = _MathHelper.getOutputImageDims(imageWidth, imageHeight,
                                              localReceptiveFieldWidth, localReceptiveFieldHeight,
                                              stride)

        # wraps the kernel in a [] and then duplicates the array for NxM
        kernel = np.repeat(kernel[:, :, None, :, :], x, axis=2)
        kernel = np.repeat(kernel[:, :, :, None, :, :], y, axis=3)
        subs = _MathHelper.getLocalReceptiveFields(matrix, stride, x, y,
                                            localReceptiveFieldWidth, localReceptiveFieldHeight)

        # multipling the kernel in the local receptive fields and summing up
        arr = subs * kernel
        conv = np.sum(subs * kernel, axis=(4, 5))

        return conv
