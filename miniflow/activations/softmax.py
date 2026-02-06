import numpy as np
from miniflow.layers.layer import Layer


class Softmax(Layer):
    def forward(self, input_data, training=True):
        tmp = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = tmp / np.sum(tmp, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient):
        n = np.size(self.output)
        return np.dot((np.eye(n) - self.output.T) * self.output, output_gradient)
