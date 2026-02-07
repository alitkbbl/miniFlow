# miniflow/layers/batchnorm.py
import numpy as np
from .layer import Layer


class BatchNormalization(Layer):
    def __init__(self, momentum=0.99, epsilon=1e-5, input_dim=None):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.input_dim = input_dim

        self.gamma = None  # Scale
        self.beta = None  # Shift

        self.dgamma = None
        self.dbeta = None
        self.weights = None
        self.biases = None
        self.dweights = None
        self.dbiases = None

        self.running_mean = None
        self.running_var = None
