"""Main module for (1) generating the SMT formula (2) solve the formula"""
import pickle

from z3 import unsat, unknown

from nnsynth.common import sanity
from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.models import OutputConstraint
from nnsynth.common.properties import KeepContextProperty, DeltaRobustnessProperty
from nnsynth.common.sanity import xor_dataset_sanity_check
from nnsynth.datasets import XorDataset
from nnsynth.formula_generator import FormulaGenerator
from nnsynth.neural_net import get_params, create_skorch_net, print_params, get_num_layers
from nnsynth.weights_selector import WeightsSelector
from nnsynth.z3_context_manager import Z3ContextManager


def main(args):
    # main flow

    # load dataset
    if args.load_dataset:
        dataset = XorDataset.from_pickle(args.load_dataset)
    else:
        exit(1)

    input_size = dataset.get_input_size()
    num_classes = dataset.get_output_size()
    dataset.filter_data(args.eval_set)

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

    # formulate in SMT via z3py
    coefs, intercepts = get_params(net)
    generator = FormulaGenerator(coefs=coefs, intercepts=intercepts, input_size=input_size,
                                 output_size=num_classes, num_layers=num_layers)
    checked_property = [
        DeltaRobustnessProperty(input_size=input_size, output_size=num_classes, desired_output=1,
                                coordinate=args.pr_coordinate, delta=args.pr_delta,
                                output_constraint_type=OutputConstraint.Max)
        ]

    weights_selector = WeightsSelector(input_size=input_size, hidden_size=(4,),
                                       output_size=num_classes, delta=args.ws_delta)

    # keep context (original NN representation)
    eval_set = dataset.get_evaluate_set(net, args.eval_set, args.eval_set_type, args.limit_eval_set)
    sanity.print_eval_set(eval_set)
    keep_ctx_property = KeepContextProperty(eval_set)

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
    thresholds = reversed([i for i in range(1, 10)])

    for weight_tuple in _comb_tuples:
        weights_selector.reset_selected_weights()
        if weight_tuple[0] == 'w':
            weights_selector.select_weight(weight_tuple[1], weight_tuple[2], weight_tuple[3])
        elif weight_tuple[0] == 'b':
            weights_selector.select_bias(weight_tuple[1], weight_tuple[2])

        for threshold in thresholds:
            keep_ctx_property.set_threshold(threshold)

            generator.generate_formula(checked_property, weights_selector, keep_ctx_property)

            z3_mgr = Z3ContextManager()
            z3_mgr.add_formula_from_memory(generator.get_goal())
            exit(0)
            # TODO: this call should invoke PBS job
            #send_pbs_job(...)

    # TODO: think of a way to organize the results (including configurations)


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)
