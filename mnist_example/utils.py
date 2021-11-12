from skimage.transform import rescale
from sklearn.model_selection import train_test_split
from sklearn import metrics
from joblib import dump, load
import numpy as np
from sklearn import svm
import os


def preprocess(images, rescale_factor):
    resized_images = []
    for d in images:
        resized_images.append(rescale(d, rescale_factor, anti_aliasing=False))
    return resized_images


def create_splits(data, targets, test_size, valid_size):
    X_train, X_test_valid, y_train, y_test_valid = train_test_split(
        data, targets, test_size=test_size + valid_size, shuffle=False
    )

    X_test, X_valid, y_test, y_valid = train_test_split(
        X_test_valid,
        y_test_valid,
        test_size=valid_size / (test_size + valid_size),
        shuffle=False,
    )
    return X_train, X_test, X_valid, y_train, y_test, y_valid


def test_model(best_model_path, X, y):
    clf = load(best_model_path)
    metrics = test(clf, X, y)
    return metrics


def test(clf, X, y):
    predicted = clf.predict(X)
    acc = metrics.accuracy_score(y_pred=predicted, y_true=y)
    f1 = metrics.f1_score(y_pred=predicted, y_true=y, average="macro")

    return {"acc": acc, "f1": f1}


def get_random_acc(y):
    return max(np.bincount(y)) / len(y)


def run_classification_experiment(classifier, X_train, y_train, X_valid, y_valid, gamma, output_model_file, skip_dummy=True):
    random_val_acc = get_random_acc(y_valid)
    # Create a classifier: a support vector classifier
    clf = classifier(gamma=gamma)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    # Predict the value of the digit on the validation subset
    metrics_valid = test(clf, X_valid, y_valid)
    # we will ensure to throw away some of the models that yield random-like performance.
    if skip_dummy and metrics_valid["acc"] < random_val_acc:
        print("Skipping for {}".format(gamma))
        return None
    
    output_folder = os.path.dirname(output_model_file)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    dump(clf, output_model_file)
    return metrics_valid
