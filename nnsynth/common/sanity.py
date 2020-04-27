"""Implementations of sanity checks for main module"""
import torch


def xor_dataset_sanity_check(net):
    test = torch.Tensor(
        [[10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, -10.0],
        [10.0, -10.0]]
    )

    return net.predict_proba(test)


def print_eval_set(eval_set):
    print("Eval set")
    X, y = eval_set
    for i, sample in enumerate(list(zip(X, y))):
        X, y = sample[0], sample[1]
        print("Sample {} - X: {}, y: {}".format(i, X, y))
