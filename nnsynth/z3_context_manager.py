import subprocess
from collections import OrderedDict
from pathlib import Path

from z3 import Solver, Goal, sat, parse_smt2_file, parse_smt2_string


class Z3ContextManager:
    def __init__(self):

        # model result
        self.result = None
        self.model = None
        self.model_mapping = OrderedDict()

        self.solver = Solver()
        self.goal = Goal()

    def add_formula_from_memory(self, goal: Goal):
        """Add formula to z3 from memory (after producing goal with generator)"""
        self.solver.add(goal)

    def add_formula_from_disk(self, formula_path: str = 'formula.nn'):
        """Add formula to z3 from disk"""
        formula = None
        with open(formula_path, 'r') as handle:
            formula = handle.read()

        z3_formula = parse_smt2_string(formula)
        self.solver.add(z3_formula)

    def solve(self):
        self.result = self.solver.check()

    # def solve(self, ssh_server):
    #     """Solve remotely on Tamnun"""
    #     FILE = Path('.').parent / ''
    #     subprocess.run(["scp", FILE, "USER@SERVER:PATH"])

    def get_result(self):
        """Return 'z3.sat', 'z3.unsat' or 'z3.unknown'"""
        return self.result

    def get_model_mapping(self, z3_weight_variables, original_weight_values):
        if self.result != sat:
            raise Exception("Cannot return model mapping for non-sat formula")

        self.model = self.solver.model()
        # evaluate searched variables
        searched_weights_keys = z3_weight_variables.keys()
        for weight_key in searched_weights_keys:
            weight_value = z3_weight_variables[weight_key]
            if self.model[weight_value] is not None:
                w_optim = float(self.model[weight_value].numerator_as_long()
                                / self.model[weight_value].denominator_as_long())
                w_orig = original_weight_values[weight_key]

                self.model_mapping[weight_key] = (w_optim, w_orig)

        return self.model_mapping

    def model_mapping_sanity_check(self):
        # print original and optimized weight values: must have model mapping
        for key, value in self.model_mapping.items():
            w_optim, w_orig = value
            print("%s approx: %.6f, orig: %.6f, diff: %.12f" %
                  (key, w_optim, w_orig, abs(w_orig - w_optim)))

    def save_formula_to_disk(self, filename='check.smt2'):
        with open(filename, 'w') as handle:
            handle.write(self.solver.sexpr())
