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
    def getLocalReceptiveFields(
            matrix,
            stride,
            imageWidth,
            imageHeight,
            lrfWidth,
            lrfHeight
    ):
        s1, s2, s3, s4, s5 = matrix.strides

        numberOfLocalReceptiveFields = \
            _MathHelper.getOutputImageDims(imageWidth, imageHeight, lrfWidth, lrfHeight, stride)[:] * \
            matrix.shape[1]

        output_shape = (
            matrix.shape[0],  # mini batch size
            matrix.shape[1],  # number of filters
            numberOfLocalReceptiveFields,
            lrfWidth,
            lrfHeight
        )

        strides = (
            s1,
            s2,
            s3,
            s4 * stride,
            s5 * stride
        )

        return np.lib.stride_tricks.as_strided(matrix, output_shape, strides=strides)

    @staticmethod
    def conv5D(self, matrix, kernel, stride=1):
        # needed variables
        imageWidth, imageHeight = matrix.shape[-2:]
        localReceptiveFieldWidth, localReceptiveFieldHeight = kernel.shape[-2:]
        numberOfLocalReceptiveFields = \
            (1 + (imageWidth - localReceptiveFieldWidth) // stride) * \
            (1 + (imageHeight - localReceptiveFieldHeight) // stride) * \
            matrix.shape[1]

        # wraps the kernel in a [] and then duplicates the array for the size of the number of local receptive fields
        kernel = np.repeat(kernel[:, :, None, :, :], numberOfLocalReceptiveFields, axis=2)

        subs = _MathHelper.getLocalReceptiveFields(matrix, stride, imageWidth, imageHeight,
                                            localReceptiveFieldWidth, localReceptiveFieldHeight)
        arr = subs * kernel
        # multipling the kernel in the local receptive fields and summing up
        conv = np.sum(subs * kernel, axis=(3, 4))

        return conv
