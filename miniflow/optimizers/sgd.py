import numpy as np
from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}

    def update(self, layer):
        if not hasattr(layer, 'weights'):
            return

        layer_id = id(layer)
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {
                'w': np.zeros_like(layer.weights),
                'b': np.zeros_like(layer.biases)
            }

        v_w = self.velocities[layer_id]['w']
        v_b = self.velocities[layer_id]['b']

        # momentum:
        # v_new = (momentum * v_old) - (lr * gradient)
        # w_new = w_old + v_new

        v_w_new = (self.momentum * v_w) - (self.learning_rate * layer.dweights)
        v_b_new = (self.momentum * v_b) - (self.learning_rate * layer.dbiases)

        layer.weights += v_w_new
        layer.biases += v_b_new

        self.velocities[layer_id]['w'] = v_w_new
        self.velocities[layer_id]['b'] = v_b_new
