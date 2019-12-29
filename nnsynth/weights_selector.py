from nnsynth.common.formats import Formats


class WeightsSelector:
    def __init__(self, input_size: int, hidden_size: tuple, output_size: int, delta: float = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # whether to add constraints for bounding the variable in a delta neighbourhood
        self.delta = delta

        # unify layers size [input, hidden1, hidden2..., output]
        self.hidden_sizes = [i for i in self.hidden_size]
        self.layers = [self.input_size] + self.hidden_sizes + [self.output_size]

        self.selected_weights = []

        self.bias_fmt = Formats.bias_fmt
        self.output_fmt = Formats.output_fmt
        self.weight_fmt = Formats.weight_fmt

    def get_selected_weights(self):
        """Return unique selected weights, in addition removes duplicate keys"""
        return sorted(set(self.selected_weights))

    def get_delta(self):
        """Returns None if no bounding is wanted; otherwise add bounds for each
        free weight as follows: original_value - delta <= variable <= original_value + delta"""
        return self.delta

    def select_layer(self, layer: int):
        """Layer == 1 means the first hidden layer"""
        # all neurons in layer
        if layer <= len(self.layers):
            num_of_neurons = self.layers[layer]
            for neuron in range(1, num_of_neurons+1):
                self.select_neuron(layer, neuron)

    def select_neuron(self, layer: int, neuron: int):
        # all weights in neuron
        previous_layer_size = self.layers[layer-1]
        if layer <= len(self.layers) and neuron <= self.layers[layer]:
            for w_id in range(1, previous_layer_size+1):
                self.select_weight(layer, neuron, w_id)

            self.select_bias(layer, neuron)

    def select_weight(self, layer: int, neuron: int, weight: int):
        # specific weight in neuron
        previous_layer_size = self.layers[layer-1]
        if layer <= len(self.layers) and neuron <= self.layers[layer] and weight <= previous_layer_size:
            self.selected_weights.append(self.weight_fmt % (layer, neuron, weight))

    def select_bias(self, layer: int, neuron: int):
        # bias of specific neuron
        if layer <= len(self.layers) and neuron <= self.layers[layer]:
            self.selected_weights.append(self.bias_fmt % (layer, neuron))
