import argparse


class ArgumentsParser:
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
