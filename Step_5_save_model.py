#
# Copyright (c) Microsoft Corporation and contributors. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import json


def save_model(model, labels, filename):
    model.save(filename + ".h5")
    with open(filename + ".json", "w") as f:
        json.dump(labels, f)
