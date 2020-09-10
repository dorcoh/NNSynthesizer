"""Implementations of sanity checks for main module"""
import torch
import numpy as np


def xor_dataset_sanity_check(net):
    test = torch.Tensor(
        [[10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, -10.0],
        [10.0, -10.0]]
    )

    return pred(net, test)


def pred(net, test_data):
    y_pred = net.predict_proba(test_data)
    y_pred_labels = np.argmax(y_pred, axis=1)
    return y_pred, y_pred_labels


def print_eval_set(eval_set):
    print("Eval set")
    X, y = eval_set
    for i, sample in enumerate(list(zip(X, y))):
        X, y = sample[0], sample[1]
        print("Sample {} - X: {}, y: {}".format(i, X, y))
