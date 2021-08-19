import logging
import sys
import unittest

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.properties import KeepContextType, KeepContextProperty, EnforceGridSoftProperty
from nnsynth.datasets import Dataset
from nnsynth.evaluate import EvaluateDecisionBoundary, build_exp_docstring
from nnsynth.neural_net import ModularClassificationNet, ClassificationNet, create_skorch_net, get_n_params

def load_nn(args):
    input_size = 2
    num_classes = 2
    net_class = ModularClassificationNet if args.modular_nn else ClassificationNet
    net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size, num_classes=num_classes,
                            epochs=args.epochs, learning_rate=args.learning_rate, random_seed=args.random_seed,
                            init=args.load_nn is not None, net_class=net_class)
    # train / load NN
    if args.load_nn:
        net.load_params(args.load_nn)

    return net

class PlotGridTestCase(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        args = ArgumentsParser.parser.parse_args()
        vars(args).update({'heuristic': KeepContextType.GRID.value})
        vars(args).update({'meshgrid_stepsize': 2.5})
        vars(args).update({'load_nn': 'resources/plot_grid/model-xor-bad.pkl'})
        original_net = load_nn(args)
        vars(args).update({'load_nn': 'resources/plot_grid/model-fixed-xor-bad.pkl'})
        fixed_net = load_nn(args)
        dataset = Dataset.from_pickle('resources/plot_grid/xor-bad-dataset.pickle')

        self.evaluator = EvaluateDecisionBoundary(original_net, fixed_net, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                             contourf_levels=args.contourf_levels, save_plot=False)

        # docstring for the plot
        self.exp_name, self.details = build_exp_docstring(args=args, num_constraints=None,
                                                time_took=None,
                                                net_params=None,
                                                net_free_params=None)

        self.patches, self.patches_labels = \
            EnforceGridSoftProperty.load_saved_patches('resources/plot_grid/keep_context_grid_patches_xor-bad.pickle')

    def test_something(self):
        self.evaluator.multi_plot_with_patches(patches=self.patches, patches_labels=self.patches_labels,
                                               exp_name=self.exp_name, sub_name=self.details)


if __name__ == '__main__':
    unittest.main()
