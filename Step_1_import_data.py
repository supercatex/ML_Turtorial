#
# Copyright (c) Microsoft Corporation and contributors. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import os
import matplotlib.pyplot as plt
import matplotlib.image as Image
import numpy as np
import random
from console_progressbar import ProgressBar


def import_data(data_directory):
    # Define X is features, y is label.
    X = []
    y = []

    # Define data-set directory.
    dir_labels = data_directory

    # We use the folder name as a label name.
    labels = os.listdir(dir_labels)

    # Search all folder in data-set directory.
    for i, label in enumerate(labels):

        # Get all image name in each folder.
        dir_images = dir_labels + os.sep + label
        image_names = os.listdir(dir_images)

        # Create a new progress bar.
        progress_bar = ProgressBar(
            total=len(image_names),
            prefix="Label(%s):" % label,
            suffix="%d/%d" % (i+1, len(labels)),
            length=50
        )

        # Search all image in each folder.
        for j, image_name in enumerate(image_names):

            # Read image from file.
            filename = dir_images + os.sep + image_name
            image = Image.imread(filename)

            # Add into X, y.
            X.append(image)
            y.append(i)

            # Update progress bar.
            progress_bar.print_progress_bar(j + 1)

    # Shuffle data.
    data = list(zip(X, y))
    random.shuffle(data)
    X, y = zip(*data)

    # Final X, y.
    X = np.array(X)
    y = np.array(y)

    return X, y, labels


if __name__ == "__main__":
    X, y, labels = import_data("./data/mnistasjpg/trainingSet")

    # Display some samples.
    rows = 3
    cols = 5
    fig = plt.figure("Samples", figsize=(10, 7))
    for i in range(rows * cols):
        r = random.randint(0, len(X))
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title("Label index: " + y[r])
        plt.imshow(X[r], cmap=plt.cm.gray)
    plt.show()
