import numpy as np
from loss import Loss

class MSE(Loss):
    def calculate(self, output, y_true):
        # Mean Squared Error
        return np.mean(np.power(y_true - output, 2))

    def gradient(self, output, y_true):
        # 2/N * (output - true)
        return 2 * (output - y_true) / y_true.size
