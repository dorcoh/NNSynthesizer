from z3 import sat

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
    dataset = XorDataset(center=args.center, std=args.std, samples=args.dataset_size,
                         test_size=args.test_size, random_seed=args.random_seed)

    X_train, y_train, X_test, y_test = dataset.get_splitted_data()

    input_size = dataset.get_input_size()
    num_classes = dataset.get_output_size()

    # train NN
    net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size,
                            num_classes=num_classes, learning_rate=args.learning_rate,
                            epochs=args.epochs, random_seed=args.random_seed)
    num_layers = get_num_layers(net)
    net.fit(X_train, y_train)

    # plot decision boundary
    evaluator = EvaluateDecisionBoundary(net, net, dataset, args.meshgrid_stepsize, args.contourf_levels,
                                         args.save_plot)
    evaluator.plot()
    print_params(net)

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
    weights_selector = WeightsSelector(input_size=input_size, hidden_size=(8,),
                                       output_size=num_classes, delta=args.ws_delta)
    weights_selector.select_neuron(layer=2, neuron=1)
    weights_selector.select_bias(layer=2, neuron=2)

    # keep context (original NN representation)
    eval_set = dataset.get_evaluate_set(net, args.eval_set, args.eval_set_type)
    keep_ctx_property = KeepContextProperty(eval_set)

    generator.generate_formula(weights_selector, checked_property, keep_ctx_property)

    z3_mgr = Z3ContextManager(generator.get_optimize_weights(), generator.get_weight_values(),
                              generator.get_variables())
    z3_mgr.add_formula_to_z3(generator.get_goal())
    z3_mgr.solve()
    res = z3_mgr.get_result()

    # exit if not sat
    if not (res == sat):
        print("Stopped with result: " + str(res))
        return 1

    model_mapping = z3_mgr.get_model_mapping()
    if model_mapping is not None:
        with open('model_mapping', 'w') as handle:
            handle.write(str(model_mapping))

    print(xor_dataset_sanity_check(net))

    # set new params and plot decision boundary
    fixed_net = set_params(net, model_mapping)
    evaluator = EvaluateDecisionBoundary(net, fixed_net, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                         contourf_levels=args.contourf_levels, save_plot=args.save_plot)
    evaluator.plot('fixed_decision_boundary')

    print(xor_dataset_sanity_check(fixed_net))


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)
