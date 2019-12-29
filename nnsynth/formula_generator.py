import operator
from collections import OrderedDict
from copy import deepcopy
from typing import List

from z3 import Real, Product, Sum, And, Goal, Solver, ForAll, sat, If, Const, \
    RealSort, unsat

from nnsynth.common.formats import Formats
from nnsynth.weights_selector import WeightsSelector
from nnsynth.common.models import InputImpliesOutputProperty


class FormulaGenerator:
    def __init__(self, coefs, intercepts, input_size, output_size, hidden_size, num_layers):
        self.coefs = coefs
        self.intercepts = intercepts
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size  # TODO: change to tuple (e.g., (100,50,)) to support multiple layers
        self.num_layers = num_layers

        self.variables = OrderedDict()
        self.constraints = OrderedDict()
        self.weight_values = OrderedDict()
        self.weight_values_copy = OrderedDict()
        self.optimize_weights = {}
        self.model_mapping = {}

        # z3 goal
        self.goal = Goal()
        self.model = None

        # variables and constraints formats
        self.input_fmt = Formats.input_fmt
        self.input_id_fmt = Formats.input_id_fmt
        self.neuron_z_fmt = Formats.neuron_z_fmt
        self.neuron_a_fmt = Formats.neuron_a_fmt
        self.neuron_a_fmt_same_layer = Formats.neuron_a_fmt_same_layer
        self.bias_fmt = Formats.bias_fmt
        self.output_fmt = Formats.output_fmt
        self.weight_fmt = Formats.weight_fmt
        self.weight_fmt_same_layer = Formats.weight_fmt_same_layer

    def generate_formula(self, weights_selector: WeightsSelector, checked_property=List[InputImpliesOutputProperty]):
        # define inputs variables
        for input_sz in range(1, self.input_size + 1):
            input_format = self.input_fmt % input_sz
            self.variables[input_format] = Real(input_format)

        # define neuron,relu,weight and bias variables
        for cur_layer, weights_layers in enumerate(self.coefs, start=1):
            for cur_neuron, weights_neurons in enumerate(weights_layers, start=1):
                neuron_format = self.neuron_z_fmt % (cur_layer, cur_neuron)
                relu_format = self.neuron_a_fmt % (cur_layer, cur_neuron)
                bias_format = self.bias_fmt % (cur_layer, cur_neuron)

                self.variables[neuron_format] = Real(neuron_format)
                self.variables[relu_format] = Real(relu_format)

                self.weight_values[bias_format] = self.intercepts[cur_layer - 1][cur_neuron - 1]
                for cur_weight, weight in enumerate(weights_neurons, start=1):
                    weight_format = self.weight_fmt % (cur_layer, cur_neuron, cur_weight)
                    self.weight_values[weight_format] = weight
                    if cur_layer == self.num_layers:
                        out_format = self.output_fmt % cur_neuron
                        self.variables[out_format] = Real(out_format)

        # keep original weights (before overriding them)
        # TODO: paramtetrize
        free_weight_format = [key for key in self.weight_values.keys() if ('weight_2_1' in key or 'bias_2' in key)]
        # free_weight_format = weights_selector.get_selected_weights()
        self.weight_values_copy = deepcopy(self.weight_values)
        self.add_weights_to_search(free_weight_format)

        # define neurons constraints
        for cur_layer, weights_layers in enumerate(self.coefs, start=1):
            for cur_neuron, weights_neurons in enumerate(weights_layers, start=1):
                self.declare_neurons_values(cur_layer, cur_neuron)

        constraints = []
        for constraint_list in self.constraints.values():
            for constraint in constraint_list:
                constraints.append(constraint)

        # add properties
        for property_constraint in checked_property:
            property_constraint.set_variables_dict(self.variables)
            constraints.append(property_constraint.get_property_constraint())

        input_vars = [value for key, value in self.variables.items() if self.input_id_fmt in key]

        main_clause = ForAll(
                input_vars,
                And(constraints)
            )

        self.goal.add(main_clause)

    def declare_neurons_values(self, cur_layer, cur_neuron):
        """Declare actual constraints for the network architecture"""
        if cur_layer < self.num_layers:
            # intermediate layers
            neuron_format = self.neuron_a_fmt % (cur_layer, cur_neuron)
        else:
            # output layer
            neuron_format = self.output_fmt % cur_neuron

        # collect previous neurons
        if cur_layer == 1:
            previous_neurons = [value for key, value in self.variables.items() if self.input_id_fmt in key]
        else:
            previous_neurons = [value for key, value in self.variables.items() if
                                self.neuron_a_fmt_same_layer % (cur_layer - 1) in key]

        # define current neuron weighted sum
        neuron_weights_values = [value for key, value in self.weight_values.items() if
                                 self.weight_fmt_same_layer % (cur_layer, cur_neuron) in key]
        neuron_bias_value = self.weight_values[self.bias_fmt % (cur_layer, cur_neuron)]

        products = [Product(w, i) for w, i in zip(previous_neurons, neuron_weights_values)]
        neuron_var = products + [neuron_bias_value]

        if cur_layer < self.num_layers:
            # use only one variable to represent a neuron
            self.variables[neuron_format] = If(Sum(neuron_var) >= 0, Sum(neuron_var), 0)
        else:
            # output
            self.variables[neuron_format] = Sum(neuron_var)

    def add_weights_to_search(self, weight_formats, delta=1):
        """Adds weight to variables dict and overrides weight_values dict with z3 variable,
        in addition this method adds constraints to keep the searched weights in a certain range"""
        for weight_format in weight_formats:
            self.variables[weight_format] = Const(weight_format, RealSort())

            # add constraint to keep new variable (weight) close to new one
            lower_bound = self.weight_values[weight_format] - delta
            upper_bound = self.weight_values[weight_format] + delta
            self.constraints[weight_format] = [self.variables[weight_format] >= lower_bound,
                                                  self.variables[weight_format] <= upper_bound]

            # override the weight values dictionary
            self.weight_values[weight_format] = self.variables[weight_format]
            # another helper dict (contains only the weights to optimize)
            self.optimize_weights[weight_format] = self.variables[weight_format]

    # TODO: put this as another class outside (z3 handler)
    def add_to_z3(self):
        self.solver = Solver()
        self.solver.add(self.goal)
        with open('check.smt2', 'w') as handle:
            handle.write(self.solver.sexpr())

    def solve_in_z3(self):
        check = self.solver.check()

        if check == sat:
            m = self.solver.model()
            try:
                # print outputs, if available
                if (m[self.variables['out_1']] is not None) and (m[self.variables['out_2']] is not None):
                    out_1_app = float(m[self.variables['out_1']].numerator_as_long()/m[self.variables['out_1']].denominator_as_long())
                    out_2_app = float(m[self.variables['out_2']].numerator_as_long()/m[self.variables['out_2']].denominator_as_long())
                    print("Out1: %.6f, Out2: %.6f" % (out_1_app, out_2_app))
            except:
                pass
            # evaluate searched variables
            searched_weights_keys = self.optimize_weights.keys()
            for w_key in searched_weights_keys:
                value = self.optimize_weights[w_key]
                if m[value] is not None:
                    w_app = float(m[value].numerator_as_long() / m[value].denominator_as_long())
                    w_orig = self.weight_values_copy[w_key]
                    print("%s approx: %.6f, orig: %.6f, diff: %.12f" %
                          (w_key, w_app, w_orig, abs(w_orig - w_app)))
                    self.model_mapping[w_key] = (w_app, w_orig)

            self.model = m
            return sat

        return unsat

    def return_model_mapping(self, solver_ret_value):
        if solver_ret_value == unsat:
            return None

        return self.model_mapping