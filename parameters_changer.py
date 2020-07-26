from collections import OrderedDict
from copy import copy

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.sanity import xor_dataset_sanity_check
from nnsynth.datasets import XorDataset
from nnsynth.evaluate import EvaluateDecisionBoundary
from nnsynth.neural_net import create_skorch_net, print_params, get_num_layers, set_params

args = ArgumentsParser.parser.parse_args()
# generate data and split
if not args.load_dataset:
    dataset = XorDataset(center=args.center, std=args.std, samples=args.dataset_size,
                         test_size=args.test_size, random_seed=args.random_seed)
    dataset.to_pickle('dataset.pkl')
else:
    dataset = XorDataset.from_pickle(args.load_dataset)

dataset.subset_data(0.01)
X_train, y_train, X_test, y_test = dataset.get_splitted_data()

input_size = dataset.get_input_size()
num_classes = dataset.get_output_size()

net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size,
                        num_classes=num_classes, learning_rate=args.learning_rate,
                        epochs=args.epochs, random_seed=args.random_seed,
                        init=args.load_nn is not None)
# train / load NN
if args.load_nn:
    net.load_params(args.load_nn)
else:
    net.fit(X_train, y_train)

print_params(net)

num_layers = get_num_layers(net)

param_val = 0.6119044423103333
# new_val = 1.5706346956308848
new_val = 0.55
delta = 2*param_val
steps = 50
step_size = float(4*param_val) / steps
start = param_val - delta
new_vals_list = [start+step_size*(i+1) for i in range(50)]
for new_val in new_vals_list:
    model_mapping = OrderedDict([('weight_1_1_1', (new_val, param_val))])
    # store original net before fix
    original_net = copy(net)

    # set new params and plot decision boundary
    fixed_net = set_params(net, model_mapping)
    # TODO: reduce the dataset size (takes time to evaluate)
    evaluator = EvaluateDecisionBoundary(original_net, fixed_net, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                         contourf_levels=args.contourf_levels, save_plot=True)
    base_dir = "parameters_changer_results/"
    evaluator.multi_plot(base_dir + 'param_orig_{}_new_{}'.format(round(param_val, 4), round(new_val, 4)),
                         sub_name='', plot_train=False, plot_test=False)

    print(xor_dataset_sanity_check(fixed_net))
