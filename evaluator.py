"""Main module for evaluating the solver results"""
import pickle
from collections import OrderedDict
from copy import copy


from nnsynth.common.arguments_handler import ArgumentsParser

from nnsynth.common.sanity import xor_dataset_sanity_check
from nnsynth.datasets import XorDataset
from nnsynth.evaluate import EvaluateDecisionBoundary
from nnsynth.neural_net import create_skorch_net, print_params, get_params, set_params, get_num_layers


def main(args):
    # main flow

    # load data
    if args.load_dataset:
        dataset = XorDataset.from_pickle(args.load_dataset)
    else:
        exit(1)

    input_size = dataset.get_input_size()
    num_classes = dataset.get_output_size()

    net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size,
                            num_classes=num_classes, learning_rate=args.learning_rate,
                            epochs=args.epochs, random_seed=args.random_seed,
                            init=args.load_nn is not None)
    # train / load NN
    if args.load_nn:
        net.load_params(args.load_nn)
    else:
        exit(1)

    print_params(net)

    num_layers = get_num_layers(net)

    # load model mapping
    model_mapping = OrderedDict()
    with open('model_mapping', 'rb') as handle:
        model_mapping = pickle.load(handle)

    print(xor_dataset_sanity_check(net))

    # store original net before fix
    original_net = copy(net)

    # set new params and plot decision boundary
    fixed_net = set_params(net, model_mapping)
    evaluator = EvaluateDecisionBoundary(original_net, fixed_net, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                         contourf_levels=args.contourf_levels, save_plot=args.save_plot)
    evaluator.multi_plot('multi_plot')

    print(xor_dataset_sanity_check(fixed_net))


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)
