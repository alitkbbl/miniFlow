import numpy as np
from loss import Loss

class BinaryCrossEntropy(Loss):
    def calculate(self, output, y_true):
        output = np.clip(output, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(output) + (1 - y_true) * np.log(1 - output))

    def gradient(self, output, y_true):
        output = np.clip(output, 1e-15, 1 - 1e-15)
        return -(y_true / output) + ((1 - y_true) / (1 - output)) / output.size
