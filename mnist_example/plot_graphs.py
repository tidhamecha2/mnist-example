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

# Import datasets, classifiers
from sklearn import datasets, svm

from skimage import data, color

import numpy as np

from joblib import dump, load

import pdb

from utils import preprocess, create_splits, test_model, run_classification_experiment

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

classifier = svm.SVC
# rescale_factors = [0.25, 0.5, 1, 2, 3]
rescale_factors = [1]
for test_size, valid_size in [(0.15, 0.15)]:
    for rescale_factor in rescale_factors:
        model_candidates = []
        for gamma in [10 ** exponent for exponent in range(-7, 0)]:
            resized_images = preprocess(
                images=digits.images, rescale_factor=rescale_factor
            )
            resized_images = np.array(resized_images)
            data = resized_images.reshape((n_samples, -1))

            X_train, X_test, X_valid, y_train, y_test, y_valid = create_splits(
                data, digits.target, test_size, valid_size
            )

            output_folder = "./models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, valid_size, rescale_factor, gamma,
            )
            output_model_file = os.path.join(output_folder, "model.joblib")
            metrics_valid = run_classification_experiment(
                classifier, X_train, y_train, X_valid, y_valid, gamma, output_model_file
            )
            if metrics_valid:
                candidate = {
                    "acc_valid": metrics_valid["acc"],
                    "f1_valid": metrics_valid["f1"],
                    "gamma": gamma,
                }
                model_candidates.append(candidate)

        max_valid_f1_model_candidate = max(
            model_candidates, key=lambda x: x["f1_valid"]
        )
        best_model_folder = "./models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
            test_size, valid_size, rescale_factor, max_valid_f1_model_candidate["gamma"]
        )
        best_model_path = os.path.join(best_model_folder, "model.joblib")
        metrics = test_model(best_model_path, X_test, y_test)

        print(
            "{}x{}\t{}\t{}:{}\t{:.3f}\t{:.3f}".format(
                resized_images[0].shape[0],
                resized_images[0].shape[1],
                max_valid_f1_model_candidate["gamma"],
                (1 - test_size) * 100,
                test_size * 100,
                metrics["acc"],
                metrics["f1"],
            )
        )
