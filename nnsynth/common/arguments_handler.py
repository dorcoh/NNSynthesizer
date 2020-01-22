import argparse


class ArgumentsParser:
    parser = argparse.ArgumentParser(description='NN Synthesizer.')
    # general args
    parser.add_argument('-rs', '--random_seed', default=42, type=int,
                        help='Random seed for creating/splitting dataset, and for training the net')
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
    # goal args
    parser.add_argument('-dw', '--ws_delta', default=0.5, type=float,
                        help='Delta value for bounding the free weights in their original neighbourhood')
    # robustness property args
    parser.add_argument('-pd', '--pr_delta', default=1, type=float,
                        help='Delta value for robustness property')
    parser.add_argument('-pc', '--pr_coordinate', nargs='+', default=(10, 10),
                        help='Property coordinates, e.g., for x1=10 x2=10: python main.py -pc 10 10')
    parser.add_argument('-ev', '--eval_set', default='train', type=str,
                        help='Add as a constraint an evaluation set (X, y), '
                             'could take: `train`, `test` or None')
    parser.add_argument('-evt', '--eval_set_type', default='ground_truth', type=str,
                        help='In case of evaluation set (eval_set != None), which target values to evaluate'
                             'Could take `ground_truth` or `predicted` ')
    # evaluation args
    parser.add_argument('-ms', '--meshgrid_stepsize', default=0.05, type=float,
                        help='Step size for dividing input space in generated meshgrid for contour plot')
    parser.add_argument('-cl', '--contourf_levels', default=50, type=int,
                        help='Number of different regions for contour lines')
    parser.add_argument('-spl', '--save_plot', default=False, type=bool,
                        help='Whether to save plots (True) or return it to calling class (False)')
