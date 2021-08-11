"""Main module for evaluating the solver results"""
import pickle
import sys
from collections import OrderedDict
from copy import copy, deepcopy
from pathlib import Path

from nnsynth.common.arguments_handler import ArgumentsParser

from nnsynth.common.sanity import xor_dataset_sanity_check
from nnsynth.common.utils import load_pickle, deserialize_exp
from nnsynth.datasets import XorDataset
from nnsynth.evaluate import EvaluateDecisionBoundary
from nnsynth.neural_net import create_skorch_net, print_params, get_params, set_params, get_num_layers


def deserialize_subexp_results(exp_path):
    return load_pickle(exp_path)


def main(args):
    # main flow

    exp = deserialize_exp(args.experiment)

    # load data
    if args.load_dataset:
        dataset = XorDataset.from_pickle(args.load_dataset)
    else:
        exit(1)

    input_size = dataset.get_input_size()
    num_classes = dataset.get_output_size()

    net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size, num_classes=num_classes,
                            epochs=args.epochs, learning_rate=args.learning_rate, random_seed=args.random_seed,
                            init=args.load_nn is not None)
    # train / load NN
    if args.load_nn:
        net.load_params(args.load_nn)
    else:
        exit(1)

    print_params(net)

    num_layers = get_num_layers(net)

    exp_path = Path('repair-results')
    sub_exp_path = exp_path / args.experiment
    # TODO: currently support only 1 sub-exp in dir

    # store original net before fix
    original_net = deepcopy(net)

    for i, sub_exp in enumerate(sub_exp_path.iterdir()):
        # general filter
        if 'result_dict' not in sub_exp.name:
            continue
        print(sub_exp.name)
        sub_exp_res_dict = deserialize_subexp_results(sub_exp.absolute())

        model_mapping = sub_exp_res_dict['mapping']
        print(model_mapping)
        print(xor_dataset_sanity_check(net))

        # set new params and plot decision boundary
        fixed_net = set_params(net, model_mapping)
        evaluator = EvaluateDecisionBoundary(original_net, fixed_net, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                             contourf_levels=args.contourf_levels, save_plot=True)

        threshold = sub_exp_res_dict['threshold']
        sub_exp_desc = "w={}, th={}, d={}".format(
            sub_exp_res_dict['weight_comb'],
            threshold,
            round(sub_exp_res_dict['distance'], ndigits=4)
        )
        print(sub_exp_desc)
        evaluator.multi_plot_with_evalset(exp['eval_set'], threshold, args.experiment, sub_exp_desc, split_sub_name=True)

        print(xor_dataset_sanity_check(fixed_net))

        if args.dev:
            sys.exit(1)


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)
