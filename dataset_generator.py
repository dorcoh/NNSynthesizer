"""Main module for generating the dataset"""
from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.datasets import XorDataset, BlobsDataset

import matplotlib.pyplot as plt


def main(args):
    # generate data and split
    dataset = XorDataset(center=args.center, std=args.std, samples=args.dataset_size,
                         test_size=args.test_size, random_seed=args.random_seed)
    dataset.to_pickle('dataset.pkl')

    plot(*dataset.get_data())
    dataset.to_pickle('check-dataset-generator.pkl')


def plot(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)
