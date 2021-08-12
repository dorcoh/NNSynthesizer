"""This module goal is to re-train an unsafe NN with unsat examples w.r.t to some property"""
import time
from copy import copy

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.sanity import pred, evaluate_test_acc
from nnsynth.datasets import Dataset, randomly_sample
from nnsynth.evaluate import EvaluateDecisionBoundary
from nnsynth.neural_net import ModularClassificationNet, ClassificationNet, create_skorch_net, get_n_params


def main(args):

    # generate data and split
    if not args.load_dataset:
        # dataset = XorDataset(center=args.center, std=args.std, samples=args.dataset_size,
        #                      test_size=args.test_size, random_seed=args.random_seed)
        # dataset.to_pickle('dataset.pkl')
        raise Exception("Must pass argument load_dataset")
    else:
        dataset = Dataset.from_pickle(args.load_dataset)


    eval_set = dataset.get_dummy_eval_set(num_properties=args.num_properties)
    X, y = eval_set

    dataset.add_samples(X, y)
    X_train, y_train, X_test, y_test = dataset.get_splitted_data()

    input_size = dataset.get_input_size()
    num_classes = dataset.get_output_size()

    # TODO: serialize the required arguments for initializing the network
    net_class = ModularClassificationNet if args.modular_nn else ClassificationNet
    net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size, num_classes=num_classes,
                            epochs=args.epochs, learning_rate=args.learning_rate, random_seed=args.random_seed,
                            init=args.load_nn is not None, net_class=net_class, freezed_weights_list=None)
    # train / load NN
    # ['w_3_2_1', 'w_3_1_2']
    if args.load_nn:
        net.load_params(args.load_nn)
    else:
        raise Exception("Must pass argument load_nn")

    # print(get_param(net, 2, 1, 0))
    # print(get_param(net, 2, 0, 1))

    print("Before")
    print(pred(net, eval_set[0]))

    X_sampled, y_sampled = randomly_sample(net, dataset)
    dataset.add_samples(X_sampled, y_sampled, dataset='sampled')

    evaluate_test_acc(net, X_test, y_test)
    evaluate_test_acc(net, X_train, y_train)
    evaluate_test_acc(net, X_sampled, y_sampled)

    original_net = copy(net)
    start_time = time.time()
    net.fit(X_train, y_train)
    time_took = time.time() - start_time
    time_took = "%.4f" % time_took

    # print(get_param(net, 2, 1, 0))
    # print(get_param(net, 2, 0, 1))

    evaluator = EvaluateDecisionBoundary(original_net, net, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                         contourf_levels=args.contourf_levels, save_plot=False)

    print("After")
    print(pred(net, eval_set[0]))

    evaluate_test_acc(net, X_test, y_test)
    evaluate_test_acc(net, X_train, y_train)
    evaluate_test_acc(net, X_sampled, y_sampled)

    # evaluator.plot(use_test=True)
    details = "Hidden: {}, # Params: {}, # Props: {}\n".format(args.hidden_size, get_n_params(net.module),
                                                                 args.num_properties) + \
              "Lasted: {} sec".format(time_took)
    evaluator.multi_plot(eval_set, name="Retrain by gradient method", sub_name=details)


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)