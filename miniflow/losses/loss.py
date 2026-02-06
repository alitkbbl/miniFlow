import numpy as np

class Loss:
    def calculate(self, output, y_true):
        raise NotImplementedError

    def gradient(self, output, y_true):
        raise NotImplementedError
