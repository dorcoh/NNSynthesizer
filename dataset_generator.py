"""Main module for generating the dataset"""
from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.datasets import XorDataset


def main(args):
    # generate data and split
    dataset = XorDataset(center=args.center, std=args.std, samples=args.dataset_size,
                         test_size=args.test_size, random_seed=args.random_seed)
    dataset.to_pickle('dataset.pkl')


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)
