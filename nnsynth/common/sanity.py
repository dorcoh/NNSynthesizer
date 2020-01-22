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
