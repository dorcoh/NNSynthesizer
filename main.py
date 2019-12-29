
from z3 import sat

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.models import OutputConstraint
from nnsynth.common.properties import RobustnessProperty
from nnsynth.datasets import XorDataset
from nnsynth.evaluate import EvaluateDecisionBoundary
from nnsynth.formula_generator import FormulaGenerator
from nnsynth.neural_net import create_skorch_net, print_params, get_params, set_params, get_num_layers

from nnsynth.weights_selector import WeightsSelector


def main():
    # get args
    args = ArgumentsParser.parser.parse_args()

    # main flow

    # generate data and split
    dataset = XorDataset(center=args.center, std=args.std, samples=args.dataset_size,
                         test_size=args.split_size)
    X_train, y_train, _, _ = dataset.get_splitted_data()
    input_size = dataset.get_input_size()
    num_classes = dataset.get_output_size()
    # train NN
    net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size,
                            num_classes=num_classes, learning_rate=args.learning_rate,
                            epochs=args.epochs)
    num_layers = get_num_layers(net)
    net.fit(X_train, y_train)

    # plot decision boundary
    evaluator = EvaluateDecisionBoundary(net, dataset)
    evaluator.plot()
    print_params(net)

    # formulate in SMT via z3py
    coefs, intercepts = get_params(net)
    generator = FormulaGenerator(coefs=coefs, intercepts=intercepts, input_size=input_size,
                                 output_size=num_classes, hidden_size=args.hidden_size,
                                 num_layers=num_layers)
    checked_property = [
        RobustnessProperty(input_size=input_size, output_size=num_classes, desired_output=1,
                           coordinate=(10, 10), delta=1, output_constraint_type=OutputConstraint.Max)
        ]

    # TODO: change hidden size type
    weights_selector = WeightsSelector(input_size=input_size, hidden_size=(8,),
                                       output_size=num_classes)
    weights_selector.select_neuron(layer=2, neuron=1)
    weights_selector.select_bias(layer=2, neuron=2)

    generator.generate_formula(weights_selector, checked_property)
    generator.add_to_z3()
    res = generator.solve_in_z3()

    # exit if not sat
    if res is not sat:
        print("Stopped with result: " + str(res))
        return 1

    model_mapping = generator.return_model_mapping(res)
    if model_mapping is not None:
        with open('model_mapping', 'w') as handle:
            handle.write(str(model_mapping))

    fixed_net = set_params(net, model_mapping)
    # plot decision boundary
    evaluator = EvaluateDecisionBoundary(fixed_net, dataset)
    evaluator.plot('fixed_decision_boundary')


if __name__ == '__main__':
    main()
