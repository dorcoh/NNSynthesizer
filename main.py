import argparse

from z3 import sat

from nnsynth.common.models import OutputConstraint
from nnsynth.common.properties import RobustnessProperty
from nnsynth.datasets import XorDataset
from nnsynth.evaluate import EvaluateDecisionBoundary
from nnsynth.formula_generator import FormulaGenerator
from nnsynth.neural_net import create_skorch_net, print_params, get_params, set_params

from nnsynth.weights_selector import WeightsSelector


def main():
    parser = argparse.ArgumentParser(description='NN Synthesizer.')
    # xor dataset args
    parser.add_argument('-d', '--dataset_size', default=1000, type=int,
                        help='Number of instances for data generation')
    parser.add_argument('-s', '--std', default=2, type=int,
                        help='Standard deviation of generated samples')
    parser.add_argument('-c', '--center', default=10, type=int,
                        help='Center coordinates, for example c=10 corresponds to genrating data '
                             'with the reference point (x,y)=(10,10)')
    parser.add_argument('-sp', '--split_size', default=0.4, type=float,
                        help='Test set percentage of generated data')
    # nn args
    parser.add_argument('-hs', '--hidden_size', default=8, type=int,
                        help='Neural net hidden layer size')
    parser.add_argument('-l', '--learning_rate', default=0.1, type=float,
                        help='Neural net training learning rate')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='Number of epochs for training the net')
    # TODO: paramterize random seed (data generation/pytorch)
    # TODO: parameterize

    # get out of dataset / net
    input_size = 2
    num_classes = 2
    num_layers = 2
    args = parser.parse_args()

    # main flow

    # generate data and split
    dataset = XorDataset(center=args.center, std=args.std, samples=args.dataset_size,
                         test_size=args.split_size)
    X_train, y_train, _, _ = dataset.get_splitted_data()
    # train NN
    net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size,
                            num_classes=num_classes, learning_rate=args.learning_rate,
                            epochs=args.epochs)
    net.fit(X_train, y_train)

    # plot decision boundary
    # evaluator = EvaluateDecisionBoundary(net, dataset)
    # evaluator.plot()
    print_params(net)

    # formulate in SMT via z3py
    coefs, intercepts = get_params(net)
    generator = FormulaGenerator(coefs=coefs, intercepts=intercepts, input_size=input_size,
                                 output_size=num_classes, hidden_size=args.hidden_size,
                                 num_layers=num_layers)
    checked_property = [
        RobustnessProperty(input_size=input_size, output_size=num_classes, desired_output=1,
                           coordinate=(10, 10), delta=9.99, output_constraint_type=OutputConstraint.Min)
        ]

    # TODO: change hidden size type
    weights_selector = WeightsSelector(input_size=input_size, hidden_size=(8,),
                                       output_size=num_classes)
    weights_selector.select_neuron(layer=2, neuron=1)
    weights_selector.select_bias(layer=2, neuron=2)
    # weights_selector.select_neuron(layer=2, neuron=2)
    # weights_selector.select_layer(layer=1)
    # weights_selector.select_weight(layer=1, neuron=2, weight=1)

    generator.generate_formula(weights_selector, checked_property)
    generator.add_to_z3()
    res = generator.solve_in_z3()

    # exit if not sat
    if res is not sat:
        print("Stopped with result: " + str(res))
        return 1

    model_mapping = generator.return_model_mapping(res)
    if model_mapping is not None:
        print(model_mapping)

    fixed_net = set_params(net, model_mapping)
    # plot decision boundary
    evaluator = EvaluateDecisionBoundary(fixed_net, dataset)
    evaluator.plot('fixed_decision_boundary')


if __name__ == '__main__':
    main()
