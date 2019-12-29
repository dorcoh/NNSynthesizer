"""
Provides:
- custom dataset, normalized and splitted into train/test
- mesh grid parameters for plotting decision boundary
"""
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


class Dataset(ABC):
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    @abstractmethod
    def make(self):
        pass

    def get_grid_params(self, step_size=0.1):
        x_min, x_max = self.X[:, 0].min() - .5, self.X[:, 0].max() + .5
        y_min, y_max = self.X[:, 1].min() - .5, self.X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                             np.arange(y_min, y_max, step_size))

        return xx, yy


class XorDataset(Dataset):
    def __init__(self, center=10, std=1, samples=1000, test_size=0.4):
        super().__init__()
        self.ctr = center
        self.std = std
        self.samples = samples
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.make()

    def process(self):
        self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)

    def make(self):
        # unpack params
        ctr = self.ctr
        std = self.std
        samples = self.samples
        np.random.seed(42)
        # prepare data points
        pos_a = (np.random.normal(ctr, std, size=samples), np.random.normal(ctr, std, size=samples))
        neg_a = (np.random.normal(-ctr, std, size=samples), np.random.normal(ctr, std, size=samples))
        pos_b = (np.random.normal(-ctr, std, size=samples), np.random.normal(-ctr, std, size=samples))
        neg_b = (np.random.normal(ctr, std, size=samples), np.random.normal(-ctr, std, size=samples))

        self.X = np.concatenate((pos_a, pos_b, neg_a, neg_b), axis=1)
        self.y = np.concatenate((np.zeros(samples * 2), np.ones(samples * 2)))

        # conversions
        self.X = self.X.T
        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.long)

        self.process()

        # split data
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=self.test_size, random_state=42)

    def get_data(self):
        return self.X, self.y

    def get_splitted_data(self):
        """Returns processed and splitted data"""
        return self.X_train, self.y_train, self.X_test, self.y_test

    def get_input_size(self):
        return self.X_train.shape[1]

    def get_output_size(self):
        # TODO
        return 2
