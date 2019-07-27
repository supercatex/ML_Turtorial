#
# Copyright (c) Microsoft Corporation and contributors. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

from keras.models import Model
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Dropout, Flatten


def create_model(input_shape, num_of_classes):
    inputs = Input(
        shape=input_shape
    )

    x = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        strides=(1, 1),
        activation="relu"
    )(inputs)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        strides=(1, 1),
        activation="relu"
    )(x)

    x = Flatten()(x)

    x = Dense(
        units=2048,
        activation="relu"
    )(x)

    x = Dense(
        units=1024,
        activation="relu"
    )(x)

    x = Dense(
        units=512,
        activation="relu"
    )(x)

    x = Dense(
        units=256,
        activation="relu"
    )(x)

    predictions = Dense(
        units=num_of_classes,
        activation="softmax"
    )(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = create_model((64, 64, 1), 120)
    model.summary()
