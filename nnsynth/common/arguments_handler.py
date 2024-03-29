import argparse


class ArgumentsParser:
    parser = argparse.ArgumentParser(description='NN Synthesizer.')
    # general args
    parser.add_argument('-cfg', '--config', type=str, default='config-example-samples.json',
                        help="Path to a json config to override the below settings.")
    parser.add_argument('-rs', '--random_seed', default=42, type=int,
                        help='Random seed for creating/splitting dataset, and for training the net')
    parser.add_argument('-cs', '--check_sat', default=False, type=bool,
                        help='Flag which indicates if to check the satisfiability of the formula, '
                             'without freeing weights')
    # xor dataset args
    parser.add_argument('-d', '--dataset_size', default=1000, type=int,
                        help='Number of instances for data generation per centroid')
    parser.add_argument('-s', '--std', default=3, type=int,
                        help='Standard deviation of generated samples')
    parser.add_argument('-c', '--center', default=10, type=int,
                        help='Center coordinates, for example c=10 corresponds to genrating data '
                             'with the reference point (x,y)=(10,10)')
    parser.add_argument('-ld', '--load_dataset', default=None, type=str,
                        help='Supply pickled dataset, optional')
    parser.add_argument('-sp', '--test_size', default=0.4, type=float,
                        help='Test set percentage of generated data')
    # nn args
    parser.add_argument('-hs', '--hidden_size', default=[4], type=int, nargs='+',
                        help='Neural net hidden layer size')
    parser.add_argument('-mnn', '--modular_nn', default=False, type=bool,
                        help='Whether to use 2 layer net or modular')
    parser.add_argument('-l', '--learning_rate', default=1e-3, type=float,
                        help='Neural net training learning rate')
    parser.add_argument('-e', '--epochs', default=100, type=int,
                        help='Number of epochs for training the net')
    parser.add_argument('-ln', '--load_nn', default=None, type=str,
                        help='Supply pickled NN, optional')
    # nn-trainer related
    parser.add_argument('-ts', '--trainer_subset', default=None, type=float, help="Perform a reduce before training.")
    parser.add_argument('-tns', '--trained_nn_suffix', default='trained_nn_suffix', type=str,
                        help="Trained NN name suffix.")
    # goal args
    parser.add_argument('-dw', '--ws_delta', default=None,
                        help='Delta value for bounding the free weights in their original neighbourhood, '
                             'takes float or None, in case of None formula generator skips adding these constraints')
    parser.add_argument('-wcf', '--exp_config_path', type=str, default="config-dev.json",
                        help="A json config for an experiment (by `repair_exp_runner.py`).")
    # robustness property args
    parser.add_argument('-pd', '--pr_delta', default=1, type=float,
                        help='Delta value for robustness property')
    parser.add_argument('-pc', '--pr_coordinate', nargs='+', default=(10, 10), type=int,
                        help='Property coordinates, e.g., for x1=10 x2=10: python main.py -pc 10 10')
    parser.add_argument('-pdo', '--pr_desired_output', default=1, type=int, help="Desired output for property (1/2)")
    parser.add_argument('-ev', '--eval_set', default='train', type=str,
                        help='Add as a constraint an evaluation set (X, y), '
                             'could take: `train`, `test` or None')
    parser.add_argument('-evt', '--eval_set_type', default='ground_truth', type=str,
                        help='In case of evaluation set (eval_set != None), which target values to evaluate'
                             'Could take `ground_truth` or `predicted` ')
    parser.add_argument('-evl', '--limit_eval_set', default=100, type=int,
                        help='In case of evaluation set (eval_set != None), this argument limits the number'
                             'of samples to collect from the set. It also controls the number of soft constraints '
                             '(if these constraints are represented as samples from eval_set)')
    parser.add_argument('-sdn', '--sampled_dataset_n', default=200, type=int, help="# of samples in sampled dataset.")
    parser.add_argument('-ngd', '--property_samples_sgd', default=200, type=int, help="# of samples to re-sample from property during SGD.")
    parser.add_argument('-soft', '--soft_constraints', default=False, type=bool,
                        help='Whether to activate soft (True) or hard (False) constraints')
    parser.add_argument('-heur', '--heuristic', default=1, type=int,
                        help="Keep context type (1 - samples, 2 - grid, 3 - Voronoi")
    parser.add_argument('-np', '--num_properties', default=1, type=int, help="Number of properties to optimize")
    # evaluation args
    parser.add_argument('-ms', '--meshgrid_stepsize', default=0.05, type=float,
                        help='Step size for dividing input space in generated meshgrid for contour plot')
    parser.add_argument('-ml', '--meshgrid_limit', default=.5, type=float,
                        help='Extra boundaries in generated meshgrid for contour plot')
    parser.add_argument('-cl', '--contourf_levels', default=50, type=int,
                        help='Number of different regions for contour lines')
    parser.add_argument('-spl', '--save_plot', default=False, type=bool,
                        help='Whether to save plots (True) or return it to calling class (False)')
    # main loop args
    parser.add_argument('-sl', '--send_single', default=False, type=bool,
                        help='Whether to send single experiment to remote server or not')
    # exp / sub exp args
    # used in exp_serializer
    parser.add_argument('-es', '--experiment', default='', type=str, help="Title of the serialized experiment")
    # used in main_loop_instances_solver
    parser.add_argument('-sef', '--sub_exp_filename', type=str, help='Sub experiment pickle filename')
    # used in instances invoker
    parser.add_argument('-sec', '--sub_exp_count', type=int, help='Cut the list of sub-experiment filenames as list[:c]')
    # dev related
    parser.add_argument('-dv', '--dev', type=bool, default=False, help="Dev related indicator")
    parser.add_argument('-enx', '--eval_nn_and_exit', type=bool, default=False, help="Main: when true eval net and exit.")
    parser.add_argument('-epnx', '--eval_nn_and_property_and_exit', type=bool, default=False, help="Show net with prop.")
    # main_loop_instances_solver related
    parser.add_argument('-sa', '--save_formula', type=bool, default=False, help="Save smt formula and continue")
    parser.add_argument('-sae', '--save_formula_and_exit', type=bool, default=False, help="Save smt and exit")
    # main_loop_instances_invoker related
    parser.add_argument('-z3', '--invoke_z3_binary', type=bool, default=False, help="Whether to invoke z3 binary (True),"
                                                                                    "or Python wrapper (False)")
    parser.add_argument('-z3t', '--z3_timeout', type=int, default=None, help="Timeout for Z3 in seconds.")
    parser.add_argument('-th', '--threshold', type=int, default=1, help="The threshold for keep context properties, "
                                                                         "in case of soft properties")
    # exp runner
    parser.add_argument('-cc', '--cache', type=bool, default=False)
    parser.add_argument('-gt', '--timestamp', type=str, default=None)
    parser.add_argument('-eid', '--exp_id', type=int, default=None)

    # inspector
    parser.add_argument('-pltn', '--plot_name', type=str, default='Some-plot')