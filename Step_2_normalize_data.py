#
# Copyright (c) Microsoft Corporation and contributors. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import numpy as np
from keras.utils import np_utils


def normalize_data(X, y=None):

    # Scale to between 0 and 1.
    X = X.astype(np.float64)
    X /= 255
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    if y is None:
        return X

    # One-hot encoding.
    y = np_utils.to_categorical(y)

    return X, y
