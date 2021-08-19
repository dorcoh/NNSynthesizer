"""Main module for training the network, requires: pickled dataset"""
import numpy as np

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.datasets import XorDataset, Dataset
from nnsynth.evaluate import EvaluateDecisionBoundary
from nnsynth.neural_net import create_skorch_net, print_params, get_num_layers, ModularClassificationNet, \
    ClassificationNet


def main(args):
    # main flow

    # generate data and split
    if args.load_dataset:
        dataset = Dataset.from_pickle(args.load_dataset)
        # dataset =
    else:
        exit(1)

    X_train, y_train, X_test, y_test = dataset.get_splitted_data()

    if args.trainer_subset:
        dataset.X_train, dataset.y_train = gen_imbalance(X_train, y_train, prob_stay=0.3)
        dataset.to_pickle('xor-bad-dataset.pickle')

    input_size = dataset.get_input_size()
    num_classes = dataset.get_output_size()

    net_class = ModularClassificationNet if args.modular_nn else ClassificationNet
    net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size, num_classes=num_classes,
                            epochs=args.epochs, learning_rate=args.learning_rate, random_seed=args.random_seed,
                            init=False, net_class=net_class)
    # train NN
    net.fit(dataset.X_train, dataset.y_train)

    # sanity
    print_params(net)
    num_layers = get_num_layers(net)
    print("Num layers: ", str(num_layers))

    evaluator = EvaluateDecisionBoundary(net, None, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                         contourf_levels=args.contourf_levels, save_plot=False,
                                         meshgrid_limit=args.meshgrid_limit)
    evaluator.plot(use_test=True)
    suffix = 'xor-bad'
    net.save_params(f_params='model-{}.pkl'.format(suffix), f_optimizer='optimizer-{}.pkl'.format(suffix),
                    f_history='history-{}.json'.format(suffix))


def gen_imbalance(X, y, lower_class=1, prob_stay=0.05):
    np.random.seed(42)
    _X, _y = [], []
    for elem in zip(X, y):
        if elem[1] == lower_class and np.random.uniform() > prob_stay:
            continue
        _X.append(elem[0])
        _y.append(elem[1])

    return np.array(_X), np.array(_y)


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)
