import unittest
from pathlib import Path

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.datasets import XorDataset

import matplotlib.pyplot as plt
import numpy as np

from nnsynth.neural_net import create_skorch_net


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        main_path = Path('.').absolute().parent
        data_path = main_path / 'good-network-data.XorDataset.pkl'
        net_path = main_path / 'model.pkl'
        dataset = XorDataset.from_pickle(data_path)
        dataset.subset_data(0.01)
        input_size = dataset.get_input_size()
        num_classes = dataset.get_output_size()
        args = ArgumentsParser.parser.parse_args()
        net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size,
                                num_classes=num_classes, learning_rate=args.learning_rate,
                                epochs=args.epochs, random_seed=args.random_seed,
                                init=True)
        net.load_params(net_path.as_posix())

        self.X_train, self.y_train = dataset.get_data()
        self.points_set = self.X_train

    def test_something(self):
        plt.scatter(self.points_set[:,0], self.points_set[:,1])
        plt.show()

    def test_draw_uniform_points(self):
        res = 10
        x = np.linspace(-50, 50, res)
        y = np.linspace(-50, 50, res)
        xx, yy = np.meshgrid(x, y)
        plt.plot(xx, yy, marker='.', color='k', linestyle='none')
        plt.show()


if __name__ == '__main__':
    unittest.main()
