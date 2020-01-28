from collections import OrderedDict
from typing import Dict

from z3 import Solver, Goal, sat

from nnsynth.common.formats import Formats


class Z3ContextManager:
    def __init__(self, optimize_weights: Dict,
                 weight_values: Dict, variables: Dict,
                 formula_fname: str = 'check.smt2'):

        self.fname = formula_fname
        self.optimize_weights = optimize_weights
        self.weight_values = weight_values
        self.variables = variables

        # model result
        self.result = None
        self.model = None
        self.model_mapping = OrderedDict()

        self.solver = Solver()
        self.goal = Goal()

    def add_formula_to_z3(self, goal: Goal, save: bool = True):
        self.solver.add(goal)
        if save:
            with open(self.fname, 'w') as handle:
                handle.write(self.solver.sexpr())

    def solve(self):
        self.result = self.solver.check()

    def get_result(self):
        """Return 'z3.sat', 'z3.unsat' or 'z3.unknown'"""
        return self.result

    def get_model_mapping(self):
        if self.result != sat:
            raise Exception("Cannot return model mapping for non-sat formula")

        self.model = self.solver.model()
        # evaluate searched variables
        searched_weights_keys = self.optimize_weights.keys()
        for weight_key in searched_weights_keys:
            weight_value = self.optimize_weights[weight_key]
            if self.model[weight_value] is not None:
                w_optim = float(self.model[weight_value].numerator_as_long()
                                / self.model[weight_value].denominator_as_long())
                w_orig = self.weight_values[weight_key]

                self.model_mapping[weight_key] = (w_optim, w_orig)

        self.output_sanity_check()
        self.model_mapping_sanity_check()

        return self.model_mapping

    def output_sanity_check(self):
        # print output values (if available)
        try:
            output_keys = [key for key, value in self.variables.items() if Formats.output_general_fmt in key]
            sorted_output_keys = sorted(output_keys)
            for key in sorted_output_keys:
                out_value = float(self.model[self.variables[key]].numerator_as_long()
                                  / self.model[self.variables[key]].denominator_as_long())
                print("%s: %.6f" % (key, out_value))
        except:
            pass

    def model_mapping_sanity_check(self):
        # print original and optimized weight values: must have model mapping
        for key, value in self.model_mapping.items():
            w_optim, w_orig = value
            print("%s approx: %.6f, orig: %.6f, diff: %.12f" %
                  (key, w_optim, w_orig, abs(w_orig - w_optim)))
