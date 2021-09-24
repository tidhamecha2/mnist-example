"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

import os
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from skimage import data, color
from skimage.transform import rescale
import numpy as np

from joblib import dump, load

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
n_samples = len(digits.images)

# rescale_factors = [0.25, 0.5, 1, 2, 3]
rescale_factors = [1]
for test_size, valid_size in [(0.15, 0.15)]:
    for rescale_factor in rescale_factors:
        model_candidates = []
        for gamma in [10 ** exponent for exponent in range(-7, 0)]:
            resized_images = []
            for d in digits.images:
                resized_images.append(rescale(d, rescale_factor, anti_aliasing=False))

            resized_images = np.array(resized_images)
            data = resized_images.reshape((n_samples, -1))

            # Create a classifier: a support vector classifier
            clf = svm.SVC(gamma=gamma)

            X_train, X_test_valid, y_train, y_test_valid = train_test_split(
                data, digits.target, test_size=test_size + valid_size, shuffle=False
            )

            X_test, X_valid, y_test, y_valid = train_test_split(
                X_test_valid,
                y_test_valid,
                test_size=valid_size / (test_size + valid_size),
                shuffle=False,
            )

            # print("Number of samples: Train:Valid:Test = {}:{}:{}".format(len(y_train),len(y_valid),len(y_test)))

            # Learn the digits on the train subset
            clf.fit(X_train, y_train)
            predicted_valid = clf.predict(X_valid)
            acc_valid = metrics.accuracy_score(y_pred=predicted_valid, y_true=y_valid)
            f1_valid = metrics.f1_score(
                y_pred=predicted_valid, y_true=y_valid, average="macro"
            )

            # we will ensure to throw away some of the models that yield random-like performance.
            if acc_valid < 0.11:
                print("Skipping for {}".format(gamma))
                continue

            candidate = {
                "acc_valid": acc_valid,
                "f1_valid": f1_valid,
                "gamma": gamma,
            }
            model_candidates.append(candidate)
            output_folder = "./models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, valid_size, rescale_factor, gamma
            )
            os.mkdir(output_folder)
            dump(clf, os.path.join(output_folder,"model.joblib"))

            
        # Predict the value of the digit on the test subset

        max_valid_f1_model_candidate = max(
            model_candidates, key=lambda x: x["f1_valid"]
        )
        best_model_folder="./models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, valid_size, rescale_factor, max_valid_f1_model_candidate['gamma']
            )
        clf = load(os.path.join(best_model_folder,"model.joblib"))
        predicted = clf.predict(X_test)

        acc = metrics.accuracy_score(y_pred=predicted, y_true=y_test)
        f1 = metrics.f1_score(y_pred=predicted, y_true=y_test, average="macro")
        print(
            "{}x{}\t{}\t{}:{}\t{:.3f}\t{:.3f}".format(
                resized_images[0].shape[0],
                resized_images[0].shape[1],
                max_valid_f1_model_candidate["gamma"],
                (1 - test_size) * 100,
                test_size * 100,
                acc,
                f1,
            )
        )

