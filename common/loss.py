import numpy as np


def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y), axis=1).reshape(y.shape[0], 1)
