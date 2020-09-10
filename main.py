"""Demonstrate the flow by invoking a repair for single network"""
import sys
from collections import OrderedDict
from copy import copy

from z3 import sat, unsat, unknown

from nnsynth.common import sanity
from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.models import OutputConstraint
from nnsynth.common.properties import DeltaRobustnessProperty, EnforceSamplesSoftProperty, EnforceGridSoftProperty, \
    EnforceSamplesHardProperty, KeepContextProperty, EnforceVoronoiSoftProperty
from nnsynth.common.sanity import xor_dataset_sanity_check, pred
from nnsynth.common.utils import save_pickle
from nnsynth.datasets import XorDataset
from nnsynth.evaluate import EvaluateDecisionBoundary
from nnsynth.formula_generator import FormulaGenerator
from nnsynth.neural_net import create_skorch_net, print_params, get_params, set_params, get_num_layers
from nnsynth.weights_selector import WeightsSelector


# TODO: complete all inner todos, get parameters out of classes,
#  make some more generic robustness proeprties (multiple quarters, etc.),
#  define some algorithm or method for choosing weights,
from nnsynth.z3_context_manager import Z3ContextManager


def main(args):
    # main flow

    # generate data and split
    if not args.load_dataset:
        dataset = XorDataset(center=args.center, std=args.std, samples=args.dataset_size,
                             test_size=args.test_size, random_seed=args.random_seed)
        dataset.to_pickle('dataset.pkl')
    else:
        dataset = XorDataset.from_pickle(args.load_dataset)

    X_train, y_train, X_test, y_test = dataset.get_splitted_data()

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
        net.fit(X_train, y_train)

    print_params(net)

    num_layers = get_num_layers(net)

    z3_mgr = Z3ContextManager()

    # formulate in SMT via z3py
    coefs, intercepts = get_params(net)
    generator = FormulaGenerator(coefs=coefs, intercepts=intercepts, input_size=input_size,
                                 output_size=num_classes, num_layers=num_layers)
    checked_property = [
        DeltaRobustnessProperty(input_size=input_size, output_size=num_classes, desired_output=1,
                                coordinate=args.pr_coordinate, delta=args.pr_delta,
                                output_constraint_type=OutputConstraint.Max)
        ]

    # TODO: wrap weights selector in some tactic generator or heuristic search,
    #  configure robustness property and weights selection
    # TODO: change hidden size type
    weights_selector = WeightsSelector(input_size=input_size, hidden_size=(4,),
                                       output_size=num_classes, delta=args.ws_delta)
    # weights_selector.select_neuron(layer=2, neuron=1)
    weights_selector.select_weight(layer=1, neuron=1, weight=1)

    if not args.dev:
        eval_set = dataset.get_evaluate_set(net, args.eval_set, args.eval_set_type, args.limit_eval_set)
    else:
        eval_set = dataset.get_dummy_eval_set()

    sanity.print_eval_set(eval_set)

    if args.soft_constraints:
        keep_ctx_property = EnforceSamplesSoftProperty()
    else:
        keep_ctx_property = EnforceSamplesHardProperty()

    keep_ctx_property.set_kwargs(**{'eval_set': eval_set, 'threshold': args.threshold})

    if args.check_sat:
        # check sat without repair
        generator.generate_formula(checked_property, None, None)
    else:
        # repair
        generator.generate_formula(checked_property, weights_selector, keep_ctx_property)

    z3_mgr.add_formula_from_memory(generator.get_goal())

    z3_mgr.save_formula_to_disk('formula-{}.smt2'.format(keep_ctx_property.get_constraints_type()))
    z3_mgr.solve()

    res = z3_mgr.get_result()

    # exit if not sat
    if (res == unsat or res == unknown) and not args.check_sat:
        print("Stopped with result: " + str(res))
        return 1

    elif args.check_sat:
        # check sat mode logic (no weights are freed, or additional constraints added)
        print("Check sat mode: formula is {}".format(str(res)))
        exit(0)

    model_mapping = z3_mgr.get_model_mapping(generator.get_z3_weight_variables(),
                                             generator.get_original_weight_values())

    # debug (for setting results of z3 solver) - can set here your params
    # model_mapping = OrderedDict([('weight_1_1_1', (0.5993294617618902, 0.6119044423103333))])
    # z3_mgr.set_model_mapping(model_mapping)

    print(z3_mgr.model_mapping_sanity_check())

    with open('main.py-model_mapping', 'w') as handle:
        handle.write(str(model_mapping))

    # store original net before fix
    original_net = copy(net)

    fixed_net = set_params(net, model_mapping)
    evaluator = EvaluateDecisionBoundary(original_net, fixed_net, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                         contourf_levels=args.contourf_levels, save_plot=False)

    # docstring for the plot
    custom_exp_name = 'soft' if args.soft_constraints else 'hard'
    custom_sub_name = 'threshold: {}\n'.format(str(args.threshold))
    custom_sub_name = custom_sub_name + ' ' + z3_mgr.model_mapping_sanity_check()

    evaluator.multi_plot_with_evalset(eval_set, name=custom_exp_name, sub_name=custom_sub_name)

    print(xor_dataset_sanity_check(net))
    print(xor_dataset_sanity_check(fixed_net))


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)
