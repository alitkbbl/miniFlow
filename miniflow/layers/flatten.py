import numpy as np
from .layer import Layer


class Flatten(Layer):
    def __init__(self, input_shape=None):
        super().__init__()
        self.input_shape = input_shape
        self.params = []
        self.input_shape_cache = None # Flatten layer has no learnable parameters

    def forward(self, input_data, training=True):
        # Cache input shape for backward pass
        self.input_shape_cache = input_data.shape

        # Convert (Batch, 28, 28) to (Batch, 784)
        return np.reshape(input_data, (input_data.shape[0], -1))

    def backward(self, output_error):
        # Reshape the error back to the original input shape
        return np.reshape(output_error, self.input_shape_cache)
