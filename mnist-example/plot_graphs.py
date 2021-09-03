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

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from skimage import data, color
from skimage.transform import rescale
import numpy as np

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

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

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
print("Image size is:")
print(digits.images[0].shape)

rescale_factors = [0.5, 1, 2]
test_sizes = [0.1, 0.2, 0.3, 0.4]
for rescale_factor in rescale_factors:
    for test_size in test_sizes:
        resized_images=[]
        for d  in digits.images:
            resized_images.append(rescale(d, rescale_factor, anti_aliasing=False))

        resized_images = np.array(resized_images)
        data = resized_images.reshape((n_samples, -1))

        # Create a classifier: a support vector classifier
        clf = svm.SVC(gamma=0.001)

        # Split data into 50% train and 50% test subsets
        X_train, X_test, y_train, y_test = train_test_split(
            data, digits.target, test_size=test_size, shuffle=False)

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted = clf.predict(X_test)

        ###############################################################################
        # Below we visualize the first 4 test samples and show their predicted
        # digit value in the title.

        _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
        for ax, image, prediction in zip(axes, X_test, predicted):
            ax.set_axis_off()
            image = image.reshape(resized_images[0].shape)
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            ax.set_title(f'Prediction: {prediction}')

        ###############################################################################
        # :func:`~sklearn.metrics.classification_report` builds a text report showing
        # the main classification metrics.

        # print(f"Classification report for classifier {clf}:\n"
        #     f"{metrics.classification_report(y_test, predicted)}\n")

        ###############################################################################
        # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
        # true digit values and the predicted digit values.

        disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
        disp.figure_.suptitle("Confusion Matrix")
        # print(f"Confusion matrix:\n{disp.confusion_matrix}")

        acc = metrics.accuracy_score(y_pred=predicted, y_true=y_test)
        f1 = metrics.f1_score(y_pred=predicted, y_true=y_test, average='macro')
        print("{}x{}\t{}:{}\t{}\t{}".format(resized_images[0].shape[0], resized_images[0].shape[1], (1-test_size)*100,test_size*100, acc,f1))

        # plt.show()
