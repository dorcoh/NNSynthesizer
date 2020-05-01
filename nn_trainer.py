"""Main module for training the network, requires: pickled dataset"""
from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.datasets import XorDataset
from nnsynth.neural_net import create_skorch_net, print_params, get_num_layers


def main(args):
    # main flow

    # generate data and split
    if args.load_dataset:
        dataset = XorDataset.from_pickle(args.load_dataset)
    else:
        exit(1)

    X_train, y_train, X_test, y_test = dataset.get_splitted_data()

    input_size = dataset.get_input_size()
    num_classes = dataset.get_output_size()

    net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size,
                            num_classes=num_classes, learning_rate=args.learning_rate,
                            epochs=args.epochs, random_seed=args.random_seed,
                            init=False)
    # train NN
    net.fit(X_train, y_train)

    # sanity
    print_params(net)
    num_layers = get_num_layers(net)
    print("Num layers: ", str(num_layers))

    net.save_params(f_params='model.pkl', f_optimizer='optimizer.pkl', f_history='history.json')


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)
