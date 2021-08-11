"""Demonstrate the flow by invoking a repair for single network"""
from copy import copy

import numpy
from z3 import unsat, unknown

from nnsynth.common import sanity
from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.models import OutputConstraint
from nnsynth.common.properties import DeltaRobustnessProperty, EnforceSamplesSoftProperty, EnforceSamplesHardProperty, \
    EnforceScoresSoftProperty, EnforceScoresHardProperty
from nnsynth.common.sanity import xor_dataset_sanity_check, evaluate_test_acc, pred
from nnsynth.datasets import XorDataset, Dataset, randomly_sample
from nnsynth.evaluate import EvaluateDecisionBoundary
from nnsynth.formula_generator import FormulaGenerator
from nnsynth.neural_net import create_skorch_net, print_params, get_params, set_params, get_num_layers, \
    ModularClassificationNet, ClassificationNet, get_n_params
from nnsynth.weights_selector import WeightsSelector


# TODO: complete all inner todos, get parameters out of classes,
#  make some more generic robustness proeprties (multiple quarters, etc.),
#  define some algorithm or method for choosing weights,
from nnsynth.z3_context_manager import Z3ContextManager


def main(args):
    # main flow

    # generate data and split
    if not args.load_dataset:
        # dataset = XorDataset(center=args.center, std=args.std, samples=args.dataset_size,
        #                      test_size=args.test_size, random_seed=args.random_seed)
        # dataset.to_pickle('dataset.pkl')
        raise Exception("Must pass argument load_dataset")
    else:
        dataset = Dataset.from_pickle(args.load_dataset)

    X_train, y_train, X_test, y_test = dataset.get_splitted_data()

    input_size = dataset.get_input_size()
    num_classes = dataset.get_output_size()

    # TODO: serialize the required arguments for initializing the network
    net_class = ModularClassificationNet if args.modular_nn else ClassificationNet
    net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size, num_classes=num_classes,
                            epochs=args.epochs, learning_rate=args.learning_rate, random_seed=args.random_seed,
                            init=args.load_nn is not None, net_class=net_class)
    # train / load NN
    if args.load_nn:
        net.load_params(args.load_nn)
    else:
        raise Exception("Must pass argument load_nn")
        # net.fit(X_train, y_train)

    if args.eval_nn_and_exit:
        evaluator = EvaluateDecisionBoundary(net, None, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                             contourf_levels=args.contourf_levels, save_plot=False)
        evaluator.plot(use_test=True)
        exit(0)

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

    weights_selector = WeightsSelector(input_size=input_size, hidden_size=args.hidden_size,
                                       output_size=num_classes, delta=args.ws_delta)
    # weights_selector.select_neuron(layer=3, neuron=1)
    # weights_selector.select_weight(layer=2, neuron=1, weight=1)
    # weights_selector.select_weight(layer=1, neuron=1, weight=1)
    # weights_selector.select_weight(layer=3, neuron=1, weight=1)
    # weights_selector.select_weight(layer=3, neuron=1, weight=2)
    weights_selector.select_weight(layer=3, neuron=2, weight=1)  # unknown for blobs-model with spec

    if not args.dev:
        eval_set = dataset.get_evaluate_set(net, args.eval_set, args.eval_set_type, args.limit_eval_set)
    else:
        eval_set = dataset.get_dummy_eval_set(args.num_properties)

    # sanity.print_eval_set(eval_set)
    print(eval_set)
    print(pred(net, eval_set[0]))

    if args.soft_constraints:
        keep_ctx_property = EnforceSamplesSoftProperty()
        # keep_ctx_property = EnforceScoresSoftProperty()
    else:
        keep_ctx_property = EnforceSamplesHardProperty()
        # keep_ctx_property = EnforceScoresHardProperty()

    # num_additional_samples = int(0.05 * y_train.shape[0])
    # soft_eval_set = [eval_set[0], eval_set[1]]
    # if num_additional_samples:
    #     X_train_subset, y_train_subset = \
    #         dataset.get_subset(X_train, y_train, num_samples=num_additional_samples, random=True)
    #     soft_eval_set[0] = numpy.append(eval_set[0], X_train_subset, axis=0)
    #     soft_eval_set[1] = numpy.append(eval_set[1], y_train_subset, axis=0)
    #     args.threshold = num_additional_samples + args.threshold

    keep_ctx_property.set_kwargs(**{'eval_set': eval_set, 'threshold': args.threshold, 'epsilon': args.epsilon})

    if args.check_sat:
        # check sat without repair
        generator.generate_formula(checked_property, None, None)
    else:
        # repair
        generator.generate_formula(checked_property, weights_selector, keep_ctx_property)
        # generator.generate_formula(checked_property, weights_selector, None)  # no keep ctx

    z3_mgr.add_formula_from_memory(generator.get_goal())

    z3_mgr.save_formula_to_disk('formula-{}.smt2'.format(keep_ctx_property.get_constraints_type()))
    time_took = z3_mgr.solve()

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
    assert model_mapping, "Empty model mapping"
    # debug (for setting results of z3 solver) - can set here your params
    # model_mapping = OrderedDict([('weight_1_1_1', (0.5993294617618902, 0.6119044423103333))])
    # z3_mgr.set_model_mapping(model_mapping)
    print(z3_mgr.model_mapping_sanity_check())

    with open('main.py-model_mapping', 'w') as handle:
        handle.write(str(model_mapping))

    # store original net before fix
    original_net = copy(net)

    # add eval set to train
    # dataset.add_samples(eval_set[0], eval_set[1], dataset='train')

    X_sampled, y_sampled = randomly_sample(net, dataset)
    dataset.add_samples(X_sampled, y_sampled, dataset='sampled')

    fixed_net = set_params(net, model_mapping)
    evaluator = EvaluateDecisionBoundary(original_net, fixed_net, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                         contourf_levels=args.contourf_levels, save_plot=False)
    fixed_net.save_params(f_params='model-fixed.pkl')

    # docstring for the plot

    custom_exp_name = 'soft' if args.soft_constraints else 'hard'
    exp_name = "Retrain by SMT - {}, Threshold: {}, # Props: {}".format(custom_exp_name, str(args.threshold),
                                                                        args.num_properties)
    details = "Result: {} \n".format(z3_mgr.model_mapping_sanity_check()) + \
              "Hidden: {}, # Params: {} \n".format(args.hidden_size, get_n_params(net.module)) + \
              "Lasted: {} sec \n".format(time_took)

    # evaluator.multi_plot_with_evalset(eval_set, name=custom_exp_name, sub_name=custom_sub_name)
    evaluator.multi_plot(eval_set, name=exp_name, sub_name=details)

    print(xor_dataset_sanity_check(original_net))
    print(xor_dataset_sanity_check(fixed_net))

    # TODO: put these inside evaluator methods
    evaluate_test_acc(original_net, X_test, y_test)
    evaluate_test_acc(fixed_net, X_test, y_test)


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)
