import numpy as np
from .activation import Activation

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

        def leaky_relu(x):
            return np.where(x > 0, x, x * self.alpha)

        def leaky_relu_prime(x):
            return np.where(x > 0, 1, self.alpha)

        super().__init__(leaky_relu, leaky_relu_prime)
