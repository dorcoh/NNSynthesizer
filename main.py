from copy import copy

from z3 import sat, unsat, unknown

from nnsynth.common import sanity
from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.models import OutputConstraint
from nnsynth.common.properties import DeltaRobustnessProperty, KeepContextProperty
from nnsynth.common.sanity import xor_dataset_sanity_check
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
    weights_selector.select_neuron(layer=2, neuron=1)
    weights_selector.select_neuron(layer=2, neuron=2)

    # keep context (original NN representation)
    eval_set = dataset.get_evaluate_set(net, args.eval_set, args.eval_set_type, 50)
    sanity.print_eval_set(eval_set)
    eval_set = XorDataset.filter_eval_set(eval_set)

    keep_ctx_property = KeepContextProperty(eval_set)

    if args.check_sat:
        generator.generate_formula(checked_property, None, None)
    else:
        generator.generate_formula(checked_property, weights_selector, keep_ctx_property)

    z3_mgr = Z3ContextManager()
    z3_mgr.add_formula_to_z3_memory(generator.get_goal())

    z3_mgr.solve()

    res = z3_mgr.get_result()

    # exit if not sat
    if (res == unsat or res == unknown) and not args.check_sat:
        print("Stopped with result: " + str(res))
        return 1

    elif args.check_sat:
        # check sat mode logic (no weights are freed, or additional constraints added)
        # TODO: decouple from here into a separated script
        print("Check sat mode: formula is {}".format(str(res)))
        exit(0)

    model_mapping = z3_mgr.get_model_mapping(generator.get_z3_weight_variables(),
                                             generator.get_original_weight_values())

    z3_mgr.model_mapping_sanity_check()

    with open('model_mapping', 'w') as handle:
        handle.write(str(model_mapping))

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
