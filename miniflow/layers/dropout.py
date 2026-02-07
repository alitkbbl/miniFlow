import numpy as np
from layer import Layer


class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        self.mask = None

    def forward(self, input_data, training=True):
        self.input = input_data

        if not training:
            return input_data

        # 0 or 1
        #  (Scaling) ->  1/(1-p)  (Inverted Dropout)
        self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape) / (1 - self.rate)

        self.output = input_data * self.mask
        return self.output

    def backward(self, output_gradient):
        # just from active
        return output_gradient * self.mask
