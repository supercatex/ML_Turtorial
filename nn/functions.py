import numpy as np


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e, axis=1).reshape(x.shape[0], 1)


def cross_entropy_error(y, t):
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
