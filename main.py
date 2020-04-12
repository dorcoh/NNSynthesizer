from copy import copy
from timeit import default_timer as timer
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
    # TODO: change hidden size type to support multiple layers
    net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size,
                            num_classes=num_classes, learning_rate=args.learning_rate,
                            epochs=args.epochs, random_seed=args.random_seed)
    num_layers = get_num_layers(net)
    net.fit(X_train, y_train)
    print_params(net)

    # formulate in SMT via z3py
    coefs, intercepts = get_params(net)
    generator = FormulaGenerator(coefs=coefs, intercepts=intercepts, input_size=input_size,
                                 output_size=num_classes, num_layers=num_layers)

    # coordinates = {(10, 10): 1, (-10, -10): 1, (-10, 10): 2, (10, -10): 2}
    coordinates = {(-10, -10): 1, (-10, 10): 2, (10, 10): 1}
    checked_property = [
        DeltaRobustnessProperty(input_size=input_size, output_size=num_classes, desired_output=output,
                                coordinate=coordinate, delta=args.pr_delta,
                                output_constraint_type=OutputConstraint.Max)
        for coordinate, output in coordinates.items()
    ]

    # TODO: wrap weights selector in a heuristic search
    weights_selector = WeightsSelector(input_size=input_size, hidden_size=(4,),
                                       output_size=num_classes, delta=args.ws_delta)
    # weights_selector.select_neuron(layer=2, neuron=1)
    # weights_selector.select_neuron(layer=2, neuron=1)
    # weights_selector.select_bias(layer=2, neuron=1)
    # weights_selector.select_bias(layer=2, neuron=2)
    # weights_selector.select_weight(layer=2, neuron=1, weight=1)
    # weights_selector.select_weight(layer=2, neuron=1, weight=2)
    # weights_selector.select_neuron(layer=2, neuron=1)
    weights_selector.select_neuron(layer=1, neuron=1)
    weights_selector.select_neuron(layer=1, neuron=2)
    # weights_selector.select_weight(layer=1, neuron=1, weight=1)
    # weights_selector.select_weight(layer=1, neuron=1, weight=2)
    # weights_selector.select_bias(layer=1, neuron=1)

    # keep context (original NN representation)
    eval_set = dataset.get_evaluate_set(net, args.eval_set, args.eval_set_type)

    # TODO: decouple soft/no soft constraints
    keep_ctx_property = KeepContextProperty(eval_set, training_percent=0.1, keep_context_threshold=0, soft=True)

    generator.generate_formula(weights_selector, checked_property,
                               keep_context_property=keep_ctx_property)

    z3_mgr = Z3ContextManager()
    z3_mgr.add_formula_to_z3(generator.get_goal())
    z3_mgr.solve()

    res = z3_mgr.get_result()

    # exit if not sat
    if not (res == sat):
        print("Stopped with result: " + str(res))
        return 1

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
    start = timer()
    args = ArgumentsParser.parser.parse_args()
    main(args)
    end = timer()
    print("Time elapsed: %.3f" % (end-start))
