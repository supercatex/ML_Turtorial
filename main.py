#
# Copyright (c) Microsoft Corporation and contributors. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from Step_1_import_data import import_data
from Step_2_normalize_data import normalize_data
from Step_3_create_model import create_model
from Step_4_training import training
from Step_5_save_model import save_model
from Step_6_load_model import load_model
from Step_7_predict import predict
import matplotlib.pyplot as plt
import os
import numpy as np


model_filename = "model"

model, labels = load_model(model_filename)

if model is None:
    X, y, labels = import_data("./data/mnistasjpg/trainingSet")
    print(X.shape, y.shape, labels)

    X, y = normalize_data(X, y)
    print(X.shape, y.shape)

    model = create_model()
    model.summary()

    model, history = training(X, y, model)

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    save_model(model, labels, model_filename)


import matplotlib.pyplot as plt
import matplotlib.image as Image

dir_test = "./data/mnistasjpg/testSet"
image_names = os.listdir(dir_test)

rows = 3
cols = 5
count = rows * cols
for i, name in enumerate(image_names):
    if i % count == 0:
        fig = plt.figure("Samples", figsize=(10, 7))

    img = Image.imread(dir_test + os.sep + name)
    X = normalize_data(np.array([img]))
    y = predict(X, model)
    ax = fig.add_subplot(rows, cols, i % count + 1)
    ax.set_title("Label index: " + str(np.argmax(y)))
    plt.imshow(img, cmap=plt.cm.gray)

    if i % count == count - 1:
        plt.show()
