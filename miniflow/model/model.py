import numpy as np
from miniflow.utils import create_batches


class History:
    """Simple class to store the training history"""

    def __init__(self):
        self.history = {}  # e.g. {'loss': [], 'val_loss': [], ...}

    def update(self, logs):
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)


class Model:
    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.loss_function = None
        self.stop_training = False  # Flag for EarlyStopping

    def add(self, layer):
        """Add a layer to the model"""
        self.layers.append(layer)

    def compile(self, optimizer, loss):
        """Configure the model for training"""
        self.optimizer = optimizer
        self.loss_function = loss

    def forward(self, x, training=True):
        """Forward pass"""
        output = x
        for layer in self.layers:
            output = layer.forward(output, training)
        return output

    def backward(self, output_gradient):
        """Backward pass (Backpropagation)"""
        gradient = output_gradient
        # Iterate from last layer to first
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def predict(self, x):
        """Prediction on new data (in inference mode)"""
        return self.forward(x, training=False)

    def fit(self, x, y, epochs=10, batch_size=32, validation_split=0.0, callbacks=None, verbose=1):
        """Main training loop"""
        history = History()

        # --- Handle validation data ---
        if validation_split > 0:
            split_idx = int(len(x) * (1 - validation_split))
            x_train, x_val = x[:split_idx], x[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            x_train, y_train = x, y
            x_val, y_val = None, None

        # --- Setup callbacks ---
        if callbacks is None:
            callbacks = []
        for cb in callbacks:
            cb.set_model(self)
            cb.on_train_begin()

        # --- Training epochs ---
        for epoch in range(epochs):
            self.stop_training = False

            train_loss = 0
            n_batches = 0

            generator = create_batches(x_train, y_train, batch_size)
            for batch_x, batch_y in generator:
                # 1. Forward
                y_pred = self.forward(batch_x, training=True)

                # 2. Compute loss
                loss = self.loss_function.forward(y_pred, batch_y)
                train_loss += loss

                # 3. Backward
                loss_grad = self.loss_function.backward(y_pred, batch_y)
                self.backward(loss_grad)

                # 4. Update weights
                for layer in self.layers:
                    self.optimizer.update(layer)

                n_batches += 1

            # Average loss per epoch
            epoch_loss = train_loss / n_batches
            logs = {'loss': epoch_loss}

            # --- Validation phase ---
            if x_val is not None:
                val_pred = self.forward(x_val, training=False)
                val_loss = self.loss_function.forward(val_pred, y_val)
                logs['val_loss'] = val_loss

            # Update history and callbacks
            history.update(logs)
            for cb in callbacks:
                cb.on_epoch_end(epoch, logs)

            # --- Progress display ---
            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f}"
                if 'val_loss' in logs:
                    msg += f" - val_loss: {logs['val_loss']:.4f}"
                print(msg)

            # --- EarlyStopping condition ---
            if self.stop_training:
                break

        return history

    def summary(self):
        """Display a summary of the model architecture"""
        print("\nModel Summary:")
        print("-" * 65)
        print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<10}")
        print("=" * 65)

        total_params = 0
        for layer in self.layers:
            layer_name = layer.__class__.__name__

            # Count parameters
            params = 0
            if hasattr(layer, 'weights') and layer.weights is not None:
                params += np.prod(layer.weights.shape)
            if hasattr(layer, 'biases') and layer.biases is not None:
                params += np.prod(layer.biases.shape)
            # Handle BatchNorm parameters
            if hasattr(layer, 'gamma') and layer.gamma is not None:
                params += np.prod(layer.gamma.shape)
            if hasattr(layer, 'beta') and layer.beta is not None:
                params += np.prod(layer.beta.shape)

            total_params += params

            # Approximate output shape (batch dimension can vary)
            out_shape = "(None, ?)"
            if hasattr(layer, 'output') and layer.output is not None:
                out_shape = str(layer.output.shape)
            elif hasattr(layer, 'output_dim'):
                out_shape = f"(None, {layer.output_dim})"

            print(f"{layer_name:<25} {out_shape:<20} {params:<10}")

        print("=" * 65)
        print(f"Total params: {total_params}")
        print("-" * 65 + "\n")
