from numpy.lib import utils
from mnist_example.utils import get_random_acc

import numpy as np

def test_rand_acc_balanced():
    y=np.array([1,1,2,2,3,3])
    assert get_random_acc(y)==1.0/3.0

def test_rand_acc_imbalanced():
    y=np.array([1,3,3,3,3])
    assert get_random_acc(y)==0.8


# TODO: write  a test case to check if model is successfully getting created or not?
# def test_model_writing():
#     1. create some data
#     2. run_classification_experiment(data, expeted-model-file)
#     assert os.path.isfile(expected-model-file)

# # TODO: write a test case to check fitting on training -- litmus test.
# def test_small_data_overfit_checking():
#     1. create a small amount of data / (digits / subsampling)
#     2. train_metrics = run_classification_experiment(train=train, valid=train)
#     assert train_metrics['acc']  > some threshold
#     assert train_metrics['f1'] > some other threshold


