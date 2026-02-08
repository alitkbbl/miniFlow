import numpy as np
from callback import Callback

class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', patience=5, min_delta=0.0):
        """
        :param monitor: Metric to monitor (usually 'val_loss')
        :param patience: Number of epochs to wait for an improvement
        :param min_delta: Minimum change to qualify as an improvement
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        # Reset state at the beginning of training
        self.wait = 0
        self.best_score = np.inf if 'loss' in self.monitor else -np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        # We assume the monitored metric is a loss (lower is better)
        # If the metric is Accuracy, the comparison logic must be reversed
        if current < self.best_score - self.min_delta:
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(f"\nEpoch {epoch+1}: early stopping")
