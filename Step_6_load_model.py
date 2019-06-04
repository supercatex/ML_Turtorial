#
# Copyright (c) Microsoft Corporation and contributors. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import keras
import json
import os


def load_model(filename):
    if os.path.exists(filename + ".h5"):
        model = keras.models.load_model(filename + ".h5")
        with open(filename + ".json", "r") as f:
            labels = json.load(f)
        return model, labels
    return None, None
