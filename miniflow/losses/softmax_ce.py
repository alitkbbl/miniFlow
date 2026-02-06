import numpy as np
from loss import Loss

class SoftmaxCrossEntropy(Loss):
    def __init__(self):
        self.softmax_output = None

    def calculate(self, logits, y_true):
        # 1. Compute Softmax
        # Subtract max value for numerical stability
        shift_logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shift_logits)
        self.softmax_output = exps / np.sum(exps, axis=1, keepdims=True)

        # 2. Compute Cross Entropy Loss
        # y_true must be one-hot encoded
        # Prevent log(0) by clipping predictions
        predictions = np.clip(self.softmax_output, 1e-15, 1 - 1e-15)

        # Compute loss only for the correct class
        return -np.mean(np.sum(y_true * np.log(predictions), axis=1))

    def gradient(self, logits, y_true):
        # Mathematical beauty:
        # The gradient of Softmax + CrossEntropy simplifies to (prediction - target)
        # Normalize by the number of samples
        return (self.softmax_output - y_true) / logits.shape[0]
