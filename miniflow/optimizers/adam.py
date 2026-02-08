import numpy as np
from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.cache = {}

    def update(self, layer):
        if not hasattr(layer, 'weights'):
            return

        layer_id = id(layer)

        if layer_id not in self.cache:
            self.cache[layer_id] = {
                'm_w': np.zeros_like(layer.weights),  # ممان اول وزن‌ها
                'v_w': np.zeros_like(layer.weights),  # ممان دوم وزن‌ها
                'm_b': np.zeros_like(layer.biases),  # ممان اول بایاس‌ها
                'v_b': np.zeros_like(layer.biases),  # ممان دوم بایاس‌ها
                't': 0
            }

        cache = self.cache[layer_id]
        cache['t'] += 1
        t = cache['t']

        cache['m_w'] = self.beta1 * cache['m_w'] + (1 - self.beta1) * layer.dweights
        cache['v_w'] = self.beta2 * cache['v_w'] + (1 - self.beta2) * (layer.dweights ** 2)

        m_w_hat = cache['m_w'] / (1 - self.beta1 ** t)
        v_w_hat = cache['v_w'] / (1 - self.beta2 ** t)

        layer.weights += -self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

        cache['m_b'] = self.beta1 * cache['m_b'] + (1 - self.beta1) * layer.dbiases
        cache['v_b'] = self.beta2 * cache['v_b'] + (1 - self.beta2) * (layer.dbiases ** 2)

        m_b_hat = cache['m_b'] / (1 - self.beta1 ** t)
        v_b_hat = cache['v_b'] / (1 - self.beta2 ** t)

        layer.biases += -self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
