"""Serializing instances of the main loop"""
import sys

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.models import OutputConstraint
from nnsynth.common.properties import EnforceSamplesSoftProperty, DeltaRobustnessProperty
from nnsynth.common.utils import deserialize_exp, serialize_main_loop_instance
from nnsynth.formula_generator import FormulaGenerator
from nnsynth.weights_selector import WeightsSelector


def generate_main_loop_instance_filename(weight_tuple, threshold, eval_set_size):
    filename = "evalsize_" + str(eval_set_size)
    filename += "_threshold_" + str(threshold)
    filename += "_param"
    for elem in weight_tuple:
        filename += "_" + str(elem)
    filename += ".pkl"
    return filename


def main(args):
    # main flow
    # TODO: merge this module into 'main_loop'? (code duplication)
    exp = deserialize_exp(args.experiment)

    # TODO: define class to make more generic
    # all combinations for 2-4-2 NN
    _comb_tuples = [
        # weights: layer, neuron, weight
        # bias: layer, neuron
        ('w', 1, 1, 1), ('w', 1, 1, 2), ('b', 1, 1),
        ('w', 1, 2, 1), ('w', 1, 2, 2), ('b', 1, 2),
        ('w', 1, 3, 1), ('w', 1, 3, 2), ('b', 1, 3),
        ('w', 1, 4, 1), ('w', 1, 4, 2), ('b', 1, 4),
        ('w', 2, 1, 1), ('w', 2, 1, 2), ('b', 2, 1),
        ('w', 2, 2, 1), ('w', 2, 2, 2), ('b', 2, 2),
    ]

    # all thresholds
    # TODO: the threshold should come from exp/params ?
    max_threshold = exp['eval_set'][1].shape[0]
    thresholds = list(reversed([i for i in range(0, max_threshold+1)]))
    # thresholds = list(reversed([i for i in range(1, 1 + 1)]))

    for weight_tuple in _comb_tuples:
        # TODO: remove limit from threshold
        for threshold in thresholds[:1]:
            filename = generate_main_loop_instance_filename(weight_tuple, threshold, exp['eval_set'][0].shape[0])
            serialize_main_loop_instance(weight_tuple, threshold, exp['eval_set'][0].shape[0], filename, args.experiment)
            if args.dev:
                sys.exit(0)


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)
