class Callback:
    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass
