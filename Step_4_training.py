#
# Copyright (c) Microsoft Corporation and contributors. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import matplotlib.pyplot as plt


def training(X, y, model):
    history = model.fit(
        x=X,
        y=y,
        validation_split=0.2,
        epochs=10,
        batch_size=128,
        verbose=1
    )

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return model, history
