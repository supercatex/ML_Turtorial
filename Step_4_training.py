#
# Copyright (c) Microsoft Corporation and contributors. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#


def training(X, y, model, epochs=10, batch_size=128):
    history = model.fit(
        x=X,
        y=y,
        validation_split=0.1,   # Separate data to training set and validation set.
        epochs=epochs,              # Repeat times.
        batch_size=batch_size,         # Each epoch data input size.
        verbose=1               # 0 = no log. 1 = log. 2 = log when epoch ending.
    )

    return model, history
