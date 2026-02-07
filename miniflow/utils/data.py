import numpy as np


def create_batches(x, y, batch_size=32, shuffle=True):

    n_samples = x.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        yield x[batch_indices], y[batch_indices]
