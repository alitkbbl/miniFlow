class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data, training=True):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError
