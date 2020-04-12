"""
Provides a trained neural network
"""
from torch import nn
import torch
import skorch


class ClassificationNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
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


def create_skorch_net(input_size, hidden_size, num_classes, epochs, learning_rate, random_seed=42):
    if random_seed is not None:
        torch.manual_seed(random_seed)
    net = skorch.NeuralNetClassifier(
        ClassificationNet(input_size, hidden_size, num_classes),
        max_epochs=epochs,
        lr=learning_rate,
        train_split=None,
        optimizer=torch.optim.Adam
    )

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


def set_params(net, model_mapping):
    # unpack the mapping and translate keys to indices
    weights = {tuple(map(int, k.split('_')[1:])): v for k, v in model_mapping.items() if 'weight' in k}
    bias = {tuple(map(int, k.split('_')[1:])): v for k, v in model_mapping.items() if 'bias' in k}

    # iterate over all weights, change only weights which have keys
    for name, param in net.module.named_parameters():
        layer_idx = int(name.split('.')[0][-1])
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

    return net


def get_num_layers(net):
    params = net.module.named_parameters()
    holder = []
    for name, param in params:
        if 'layer' in name:
            holder.append(name.split('.')[0])

    return len(list(set(holder)))


def get_predicted_tuple(net, X):
    """Returns (X, y_pred)"""
    y_pred = net.predict(X)
    return X, y_pred
