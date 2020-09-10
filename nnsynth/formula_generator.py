import pickle
from collections import OrderedDict
from copy import deepcopy, copy
from typing import Union

from z3 import Real, Product, Sum, And, Goal, ForAll, If, Const, RealSort, RealVar, RealVal

from nnsynth.common.formats import Formats
from nnsynth.common.properties import EnforceSamplesSoftProperty, EnforceGridSoftProperty, EnforceSamplesHardProperty, \
    KeepContextProperty, ConstraintType, KeepContextType
from nnsynth.weights_selector import WeightsSelector


class FormulaGenerator:
    """SMT formula generator able to encode neural networks architecture, and repair the network according to
     a certain property. It allows to choose which weights to free, and enforce a keeping context property."""

    def __init__(self, coefs: list, intercepts: list, input_size: int, output_size: int, num_layers: int):
        """
        Takes the network architecture and parameters
        :param coefs: a list of lists which contains the weight values for each neuron
        :param intercepts: a list of lists which contains the bias values for each neuron
        :param input_size: the network input size (**DEPRECATED**)
        :param output_size: the network output size (**DEPRECATED**)
        :param num_layers: number of layers (**DEPRECATED**)
        """
        self.coefs = coefs
        self.intercepts = intercepts
        self.input_size = len(coefs[0][0])  # size of first neuron of first layer
        self.output_size = len(coefs[-1])  # size of last layer
        self.num_layers = len(coefs)

        self.variables = OrderedDict()
        self.soft_constraints_variables = []
        self.original_weight_values = OrderedDict()
        self.weight_values_copy = OrderedDict()  # TODO: check if redundant
        self.z3_weight_variables = {}
        self.model_mapping = {}

        # z3 goal
        self.goal = Goal()
        self.model = None
        self.constraints = []
        self.keep_context_property = None

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

    def generate_formula(self,
                         checked_property,
                         weights_selector: Union[WeightsSelector, None],
                         keep_context_property: Union[KeepContextProperty, None] = None
                         ):
        """Generates the base structure of our formula - network architecture constraints, checked property, and
        weights we intend to search on."""

        self.declare_z3_variables()

        # keep original weights (before overriding them)
        self.weight_values_copy = deepcopy(self.original_weight_values)
        # add weights to search
        if weights_selector:
            selected_weights = weights_selector.get_selected_weights()
            self.add_weights_to_search(selected_weights)

        # define neurons constraints
        self.declare_neuron_values()

        # add checked properties
        for property_constraint in checked_property:
            property_constraint.set_variables_dict(self.variables)
            self.constraints.append(property_constraint.get_property_constraint())

        # adds (Hard) constraints to inner clause
        self.keep_context_property = keep_context_property
        self._add_pre_constraints_to_main_clause()

        # get inputs variables by their format
        self.input_vars = [value for key, value in self.variables.items() if self.input_id_fmt in key]

        self.inner_constraints_clause = And(self.constraints)

        # main property
        self.quant_clause = ForAll(
            self.input_vars,
            self.inner_constraints_clause
        )

        self.goal.add(self.quant_clause)

        # adds (Soft) constraints to goal
        self._add_post_constraints_to_goal()

    def _add_pre_constraints_to_main_clause(self):
        """Adds constraints to the main constraints clause: `self.constraints`
        Currently adds only hard keep context property constraints"""
        if self.keep_context_property is None or \
                not self.keep_context_property.get_constraints_type() == ConstraintType.HARD:
            # exit in case of empty property or non hard property
            return

        kc_constraints = self.build_keeping_context_property(self.keep_context_property)
        self.constraints += kc_constraints

    def _add_post_constraints_to_goal(self):
        """Adds (soft) constraints to the z3.Goal"""
        if self.keep_context_property is None \
                or not self.keep_context_property.get_constraints_type() == ConstraintType.SOFT:
            return

        kc_constraints = self.build_keeping_context_property(self.keep_context_property)
        self.goal.add(kc_constraints)

    def build_keeping_context_property(self, keep_context_property: KeepContextProperty) -> list:
        """Adds keeping context property to the formula
        :param keep_context_property: an instance of the desired keeping context property
        :param kwargs: additional required arguments (e.g., evaluation set, threshold)
        :return: list of keep context properties
        """
        if keep_context_property.get_constraints_type() == ConstraintType.HARD:
            keep_context_property.set_variables(self.variables)

        elif keep_context_property.get_constraints_type() == ConstraintType.SOFT:

            if keep_context_property.get_keep_context_type() == KeepContextType.SAMPLES:
                for input_value in keep_context_property.iter_soft_constraints_inputs():
                    self.declare_z3_input_variables(input_value)
                    self.declare_neuron_values()
                    self.soft_constraints_variables.append(deepcopy(self.variables))

                keep_context_property.set_variables(self.soft_constraints_variables)

            elif keep_context_property.get_keep_context_type() in (KeepContextType.GRID, KeepContextType.VORONOI):
                keep_context_property.set_variables(self.variables)

        #  challenge - the property is actually being built here, must use packages we have on Tamnun
        constraints = keep_context_property.get_property_constraint()

        return constraints

    def declare_z3_variables(self):
        """Declare the required Z3 variables for our goal"""

        self.declare_z3_input_variables()

        # define neuron,relu,weight and bias variables
        for cur_layer, weights_layers in enumerate(self.coefs, start=1):
            for cur_neuron, weights_neurons in enumerate(weights_layers, start=1):
                neuron_format = self.neuron_z_fmt % (cur_layer, cur_neuron)
                relu_format = self.neuron_a_fmt % (cur_layer, cur_neuron)
                bias_format = self.bias_fmt % (cur_layer, cur_neuron)

                self.variables[neuron_format] = Real(neuron_format)
                self.variables[relu_format] = Real(relu_format)

                self.original_weight_values[bias_format] = self.intercepts[cur_layer - 1][cur_neuron - 1]
                for cur_weight, weight in enumerate(weights_neurons, start=1):
                    weight_format = self.weight_fmt % (cur_layer, cur_neuron, cur_weight)
                    self.original_weight_values[weight_format] = weight
                    if cur_layer == self.num_layers:
                        out_format = self.output_fmt % cur_neuron
                        self.variables[out_format] = Real(out_format)

    def declare_z3_input_variables(self, input_value: float = None):
        """ Declare the required Z3 input variables for our goal
        :param input_value: In case of None input_value, the method sets its value as
        an input variable (z3 Real value expression).
        :return: nothing, modifies `self.variables`
        """
        for input_sz in range(1, self.input_size + 1):
            var_name = self.input_fmt % input_sz
            if input_value is None:
                # for main clause
                self.variables[var_name] = Real(var_name)
            else:
                # case of soft constraints - samples type
                self.variables[var_name] = RealVal(input_value[input_sz - 1])

    def declare_neuron_values(self):
        """Declare actual constraints for the network architecture"""
        for cur_layer, weights_layers in enumerate(self.coefs, start=1):
            for cur_neuron, weights_neurons in enumerate(weights_layers, start=1):
                self._declare_single_neuron_value(cur_layer, cur_neuron)

    def _declare_single_neuron_value(self, cur_layer, cur_neuron):
        """Declare a single neuron value"""
        if cur_layer < self.num_layers:
            # intermediate layers
            neuron_format = self.neuron_a_fmt % (cur_layer, cur_neuron)
        else:
            # output layer
            neuron_format = self.output_fmt % cur_neuron

        # collect previous neurons
        if cur_layer == 1:
            # input variables
            previous_neurons = [value for key, value in self.variables.items() if self.input_id_fmt in key]
        else:
            previous_neurons = [value for key, value in self.variables.items() if
                                self.neuron_a_fmt_same_layer % (cur_layer - 1) in key]

        # define current neuron weighted sum
        neuron_weights_values = [value for key, value in self.original_weight_values.items() if
                                 self.weight_fmt_same_layer % (cur_layer, cur_neuron) in key]
        neuron_bias_value = self.original_weight_values[self.bias_fmt % (cur_layer, cur_neuron)]

        products = [Product(w, i) for w, i in zip(previous_neurons, neuron_weights_values)]
        neuron_var = products + [neuron_bias_value]

        if cur_layer < self.num_layers:
            # use only one variable to represent a neuron
            self.variables[neuron_format] = If(Sum(neuron_var) >= 0, Sum(neuron_var), 0)
        else:
            # output
            self.variables[neuron_format] = Sum(neuron_var)

    def add_weights_to_search(self, weight_formats):
        """Adds weight to variables dict and overrides original_weight_values dict with z3 variable"""
        for weight_format in weight_formats:
            self.variables[weight_format] = Real(weight_format)
            # override the weight values dictionary
            self.original_weight_values[weight_format] = self.variables[weight_format]
            # another helper dict (contains only the weights to optimize)
            self.z3_weight_variables[weight_format] = self.variables[weight_format]

    def get_z3_weight_variables(self):
        if not self.z3_weight_variables:
            raise Exception("Cannot return z3_weight_variables as it's empty")

        return self.z3_weight_variables

    def get_original_weight_values(self):
        if not self.weight_values_copy:
            raise Exception("Cannot return weight_values_copy as it's empty")

        return self.weight_values_copy

    def reset_weight_variables_values(self):
        self.original_weight_values = OrderedDict()
        self.z3_weight_variables = {}

    def get_variables(self):
        if not self.variables:
            raise Exception("Cannot return variables as it's empty")

        return self.variables

    def get_goal(self):
        if not self.goal:
            raise Exception("Cannot return goal as it's empty")

        return self.goal
