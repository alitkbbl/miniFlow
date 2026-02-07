import numpy as np
from .layer import Layer


class BatchNormalization(Layer):
    def __init__(self, momentum=0.99, epsilon=1e-5, input_dim=None):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.input_dim = input_dim

        # Learnable parameters
        self.gamma = None  # Scale parameter
        self.beta = None  # Shift parameter

        # Gradients of parameters
        self.dgamma = None
        self.dbeta = None
        self.weights = None  # For optimizer compatibility (points to gamma)
        self.biases = None  # For optimizer compatibility (points to beta)
        self.dweights = None
        self.dbiases = None

        # Running statistics (used during inference/testing)
        self.running_mean = None
        self.running_var = None

    def build(self, input_shape):
        dim = input_shape[-1]
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))

        # Bind parameters to standard names for the optimizer
        self.weights = self.gamma
        self.biases = self.beta

        self.running_mean = np.zeros((1, dim))
        self.running_var = np.ones((1, dim))

    def forward(self, input_data, training=True):
        if self.gamma is None:
            self.build(input_data.shape)

        if training:
            # 1. Compute mean and variance of the current batch
            batch_mean = np.mean(input_data, axis=0, keepdims=True)
            batch_var = np.var(input_data, axis=0, keepdims=True)

            # 2. Update running (moving) statistics for inference
            self.running_mean = (
                    self.momentum * self.running_mean
                    + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                    self.momentum * self.running_var
                    + (1 - self.momentum) * batch_var
            )

            # 3. Normalize
            self.x_centered = input_data - batch_mean
            self.std_inv = 1.0 / np.sqrt(batch_var + self.epsilon)
            self.x_norm = self.x_centered * self.std_inv

            # 4. Scale and shift
            self.output = self.gamma * self.x_norm + self.beta

            # Save for backward pass
            self.batch_mean = batch_mean
            self.batch_var = batch_var

        else:
            # During inference, use running statistics
            x_norm = (
                             input_data - self.running_mean
                     ) / np.sqrt(self.running_var + self.epsilon)
            self.output = self.gamma * x_norm + self.beta

        return self.output

    def backward(self, output_gradient):
        N, D = output_gradient.shape

        # Gradients w.r.t. parameters
        self.dgamma = np.sum(
            output_gradient * self.x_norm,
            axis=0,
            keepdims=True
        )
        self.dbeta = np.sum(
            output_gradient,
            axis=0,
            keepdims=True
        )

        # Bind gradients for optimizer
        self.dweights = self.dgamma
        self.dbiases = self.dbeta

        # Gradient w.r.t. input (BatchNorm backprop – simplified formula)
        dx_norm = output_gradient * self.gamma
        dx = (1. / N) * self.std_inv * (
                N * dx_norm
                - np.sum(dx_norm, axis=0)
                - self.x_norm * np.sum(dx_norm * self.x_norm, axis=0)
        )

        return dx
