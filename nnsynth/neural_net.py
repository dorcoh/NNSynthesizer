"""
Provides a trained neural network
"""
import random
from enum import Enum
from typing import Tuple, List, Union

from skorch.callbacks import Callback
from torch import nn, tensor
import torch
import skorch


# TODO: add as identifier when adding more architectures.
class ArchitectureType(Enum):
    TwoLayers = 1


class ClassificationNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: Tuple[int], num_classes: int):
        super().__init__()
        hidden_size = hidden_size[0]
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.softmax(out)
        return out


class ModularClassificationNet(nn.Module):
    """Modular NN with this template: INPUT -> (Linear,ReLU) * N -> OUT"""
    def __init__(self, input_size: int, hidden_size: Tuple[int], num_classes: int):
        """
        Initialize a modular neural net
        :param input_size: input dimension
        :param hidden_size: tuple which represents the hidden layers and their size,
        e.g. (10,10) - 3-layer NN with 10 neurons on each layer
        :param num_classes: output dimension
        """
        super().__init__()
        self.layers = nn.ModuleList()

        number_of_mid_layers = len(hidden_size)-1

        activation = nn.ReLU()
        softmax = nn.Softmax()

        # first layer
        self.layers.append(nn.Linear(input_size, hidden_size[0]))
        self.layers.append(activation)

        for layer_num in range(number_of_mid_layers):
            # mid layers
            self.layers.append(nn.Linear(hidden_size[layer_num], hidden_size[layer_num+1]))
            self.layers.append(activation)

        # last layer
        self.layers.append(nn.Linear(hidden_size[number_of_mid_layers], num_classes))
        self.layers.append(softmax)

    def forward(self, x):
        out = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                out = layer(x)
            else:
                out = layer(out)

        return out


class FreezeWeightsCallback(Callback):
    def __init__(self, weights_list: Union[List[str], None]):
        self.weights_list = weights_list

    def on_grad_computed(self, net, named_parameters,
                         X=None, y=None, training=None, **kwargs):
        """Called once per batch after gradients have been computed but before
        an update step was performed."""
        if self.weights_list is None:
            return

        def convert_weight_name_to_indices(weight_name):
            indices = [int(x) - 1 for x in weight_name.split('_')[1:]]
            return tuple(indices)

        converted_weights_index_tuples = [convert_weight_name_to_indices(w_name) for w_name in self.weights_list]

        i = 0
        last_layer_index = len([p for p in net.module.parameters()]) - 1
        weight_bias = True  # if True - weight layer, False - bias layer
        for param_name, param in net.module.named_parameters():
            if i == last_layer_index or i % 2 == 1:
                i += 1
                continue
            layer = int(i / 2)
            weight_bias = i % 2 == 0
            mask = torch.ones(param.grad.size()).bool()
            current_weights_indices = [(x[1], x[2]) for x in converted_weights_index_tuples if x[0] == layer]
            for w_tuple in current_weights_indices:
                j, k = w_tuple
                print("Freeing weight: layer={},neuron={},weight={}".format(layer, j, k))
                mask[j][k] = False
            param.grad[mask] = 0

            i += 1


def create_skorch_net(input_size, hidden_size, num_classes, epochs, learning_rate, random_seed=42, init=False,
                      net_class=ClassificationNet, freezed_weights_list=None):
    torch.manual_seed(random_seed)
    net = skorch.NeuralNetClassifier(
        net_class(input_size, hidden_size, num_classes),
        max_epochs=epochs,
        lr=learning_rate,
        train_split=None,
        optimizer=torch.optim.Adam,
        verbose=0
        # callbacks=[
        #     FreezeWeightsCallback(freezed_weights_list)
        # ]
    )

    # required for loading stored net
    if init:
        net.initialize()

    return net


def print_params(net):
    params = net.module.named_parameters()

    for name, param in params:
        print(name, param)


def get_params(net):
    coefs, intercepts = [], []
    params = net.module.named_parameters()

    for name, param in params:
        curr = param.data.tolist()
        if 'weight' in name:
            coefs.append(curr)
        elif 'bias' in name:
            intercepts.append(curr)

    return coefs, intercepts

# TODO: add support for bias weights
def get_param(net, i, j, k):
    """Get specific parameter of network, by layer (i), neuron (j) and weight (k)"""
    _i = 0
    for param in net.module.parameters():

        if _i % 2 == 1:
            _i += 1
            continue

        layer = int(_i / 2)
        if layer == i:
            return param[j][k]

        _i += 1

def set_params(net, model_mapping):
    # unpack the mapping and translate keys to indices
    weights = {tuple(map(int, k.split('_')[1:])): v for k, v in model_mapping.items() if 'weight' in k}
    bias = {tuple(map(int, k.split('_')[1:])): v for k, v in model_mapping.items() if 'bias' in k}

    # iterate over all weights, change only weights which have keys

    i = 0
    for name, param in net.module.named_parameters():
        # e.g., name == 'layer1.weight'
        # layer_idx = int(name.split('.')[0][-1])
        layer_idx = int(i / 2) + 1
        weights_ref = None
        if 'weight' in name:
            weights_ref = weights
        elif 'bias' in name:
            weights_ref = bias

        current_layer_weights = {k: v for k, v in weights_ref.items() if k[0] == layer_idx}

        if 'weight' in name:
            for key, value in current_layer_weights.items():
                # -1 since we start keys from 1
                param.data[key[1]-1][key[2]-1] = value[0]
        elif 'bias' in name:
            for key, value in current_layer_weights.items():
                param.data[key[1]-1] = value[0]

        i += 1

    return net


def get_num_layers(net):
    params = net.module.named_parameters()
    holder = []
    for name, param in params:
        if 'layer' in name:
            holder.append(name.split('.')[0])

    return len(list(set(holder)))


def get_num_layers(hidden_size):
    return len(hidden_size)


def get_predicted_tuple(net, X):
    """Returns (X, y_pred)"""
    y_pred = net.predict(X)
    return X, y_pred


def get_predicted_score_tuple(net, X):
    """Returns (X, y (tag), score)"""
    score = net.predict_proba(X)
    y_pred = net.predict(X)

    return X, y_pred, score


def get_n_params(model):
    """Compute and return number of parameters model has"""
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp