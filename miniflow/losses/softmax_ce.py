import numpy as np
from .loss import Loss


class SoftmaxCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]

    def backward(self, y_pred, y_true):
        return (y_pred - y_true) / y_pred.shape[0]