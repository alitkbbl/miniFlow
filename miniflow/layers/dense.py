import numpy as np
from .layer import Layer


class Dense(Layer):
    def __init__(self, output_dim, input_dim=None, kernel_initializer='glorot_uniform'):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.kernel_initializer = kernel_initializer

        self.weights = None
        self.biases = None
        self.dweights = None
        self.dbiases = None

        if self.input_dim is not None:
            self.build(self.input_dim)

    def build(self, input_dim):
        self.input_dim = input_dim

        scale = 0.01

        if self.kernel_initializer == 'he':
            # ReLU/LeakyReLU
            scale = np.sqrt(2.0 / input_dim)
        elif self.kernel_initializer in ['xavier', 'glorot_uniform']:
            # Sigmoid/Tanh/Softmax
            scale = np.sqrt(1.0 / input_dim)  # یا 2.0 / (in + out)

        self.weights = np.random.randn(input_dim, self.output_dim) * scale
        self.biases = np.zeros((1, self.output_dim))

        self.is_built = True

    def forward(self, input_data, training=True):
        if self.weights is None:
            self.build(input_data.shape[1])

        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient):
        self.dweights = np.dot(self.input.T, output_gradient)

        # dB = sum(dY)
        self.dbiases = np.sum(output_gradient, axis=0, keepdims=True)

        # dX = dY * W.T
        input_gradient = np.dot(output_gradient, self.weights.T)

        return input_gradient
