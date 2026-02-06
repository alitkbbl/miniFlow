from miniflow.layers.layer import Layer


class Activation(Layer):
    def __init__(self, activation_func, activation_prime):
        super().__init__()
        self.activation = activation_func
        self.activation_prime = activation_prime

    def forward(self, input_data, training=True):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient):
        # for chain rule
        return output_gradient * self.activation_prime(self.input)
