"""
Provides:
- custom dataset, normalized and splitted into train/test
- mesh grid parameters for plotting decision boundary
"""
import pickle
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

from nnsynth.neural_net import get_predicted_tuple


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
    def __init__(self, center=10, std=1, samples=1000, test_size=0.4, random_seed=42):
        super().__init__()
        self.ctr = center
        self.std = std
        self.samples = samples
        self.test_size = test_size
        self.random_seed = random_seed
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
        np.random.seed(self.random_seed)
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

        # self.process()

        # split data
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_seed)

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

    def get_subset(self, X, y, num_samples, random):
        if random:
            mask = np.random.choice(self.y_test.shape[0], 3)
        else:
            mask = np.arange(start=0, stop=num_samples)

        return X[mask], y[mask]

    def get_test_subset(self, num_test_samples=None, random=False):
        return self.get_subset(self.X_test, self.y_test, num_test_samples, random)

    def get_train_subset(self, num_train_samples=None, random=False):
        return self.get_subset(self.X_test, self.y_test, num_train_samples, random)

    def get_evaluate_set(self, net, eval_set, eval_set_type, limit_num_samples=None):
        """Get evaluation set X, y to add later as constraints,
        In case eval_set_type is `predicted` we also need the NN object (param: net),
        if eval_set_type is `ground_truth` then net parameter is not used"""
        ret_eval_set = None
        eval_set_mapping = {'train': (self.X_train, self.get_train_subset),
                            'test': (self.X_test, self.get_test_subset)}
        if eval_set is not None:
            if eval_set_type == 'predicted':
                X = eval_set_mapping[eval_set][0]
                ret_eval_set = get_predicted_tuple(net, X)
            elif eval_set_type == 'ground_truth':
                func = eval_set_mapping[eval_set][1]
                if limit_num_samples:
                    ret_eval_set = func(limit_num_samples)
                else:
                    ret_eval_set = func()

        return ret_eval_set

    @staticmethod
    def is_noisy_sample(a: np.ndarray):
        x1, x2, y_sample = a[0], a[1], a[2]
        if x1 > 0 and x2 > 0 and y_sample == 1:
            return True
        elif x1 < 0 and x2 < 0 and y_sample == 1:
            return True
        elif x1 > 0 and x2 < 0 and y_sample == 0:
            return True
        elif x1 < 0 and x2 > 0 and y_sample == 0:
            return True
        return False

    def filter_data(self, eval_set: str):
        """Filter noisy data from self, according to desired eval_set ('train', or 'test')"""
        mask = []

        def get_mask(X, y):
            conc_arr = np.hstack((X, y.reshape(-1, 1)))
            mask = np.apply_along_axis(self.is_noisy_sample, 1, conc_arr)

            return ~mask

        if eval_set == 'train':
            mask = get_mask(self.X_train, self.y_train)
            self.X_train, self.y_train = self.X_train[mask], self.y_train[mask]
        elif eval_set == 'test':
            mask = get_mask(self.X_test, self.y_test)
            self.X_test, self.y_test = self.X_train[mask], self.y_test[mask]

    def to_pickle(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, file_path):
        with open(file_path, 'rb') as handle:
            inst = pickle.load(handle)
        if not isinstance(inst, cls):
            raise TypeError('Unpickled object is not of type {}'.format(cls))

        return inst