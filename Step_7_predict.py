#
# Copyright (c) Microsoft Corporation and contributors. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from Step_1_import_data import import_data
from Step_2_normalize_data import normalize_data


def predict(X, model):
    return model.predict(X)
