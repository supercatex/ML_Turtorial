#
# Copyright (c) Microsoft Corporation and contributors. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#


def training(X, y, model):
    history = model.fit(
        x=X,
        y=y,
        validation_split=0.2,
        epochs=10,
        batch_size=128,
        verbose=1
    )

    return model, history
