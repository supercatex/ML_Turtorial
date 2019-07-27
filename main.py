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
import cv2
import numpy as np


# Dataset: https://www.kaggle.com/scolianni/mnistasjpg

model_filename = "model"
model, labels = load_model(model_filename)
size = (50, 50)

if model is None:
    X, y, labels = import_data("./dataset/output/", size)
    print(X.shape, y.shape, labels)

    X, y = normalize_data(X, y)
    print(X.shape, y.shape)
    print(X.shape[1:])

    model = create_model(X.shape[1:], len(labels))
    model.summary()

    model, history = training(X, y, model)

    save_model(model, labels, model_filename)

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


f = open("./Macau2019/training.csv", "r")
line = f.readline()
print(line)

n = 0
t = 0
while True:
    line = f.readline()
    if len(line) == 0:
        break

    data = line.split(",")

    img = cv2.imread("./Macau2019/img/" + data[0], cv2.IMREAD_UNCHANGED)

    x1 = int(data[3])
    y1 = int(data[4])
    x2 = int(data[5])
    y2 = int(data[6])
    sign = img[y1:y2, x1:x2]
    cv2.imshow("sign", sign)
    sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)

    input = cv2.resize(sign, size)
    input = normalize_data(np.array([input]))

    py = np.argmax(predict(input, model)[0]) + 1
    ty = int(data[7])
    print(py, ty)

    if py == ty:
        t += 1
    n += 1

    cv2.waitKey(1)

print(t, n, t / n)
