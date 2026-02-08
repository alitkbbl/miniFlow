import numpy as np
from .loss import Loss


class BinaryCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (-(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))) / y_true.size
