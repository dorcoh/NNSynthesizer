import unittest
from ast import literal_eval
from pathlib import Path

from main import main


class TestNNSynthesizer(unittest.TestCase):
    def test_flow(self):
        """Tests output is equal to expected output, when solving for
        specific parameters which for the solution is known"""

        # expected files
        expected_formula = Path('resources/check.smt2')
        expected_solution = Path('resources/model_mapping.test')
        ex_decision_boundary = Path('resources/decision_boundary.png')
        ex_fixed_decision_boundary = Path('resources/fixed_decision_boundary.png')

        # generated
        generated_formula = Path('check.smt2')
        generated_solution = Path('model_mapping')
        gen_decision_boundary = Path('decision_boundary.png')
        gen_fixed_decision_boundary = Path('fixed_decision_boundary.png')

        main()

        assert expected_formula.read_bytes() == generated_formula.read_bytes()

        ex_sol_dict = literal_eval(expected_solution.read_text())
        gen_sol_dict = literal_eval(generated_solution.read_text())

        assert sorted(ex_sol_dict) == sorted(gen_sol_dict)

        assert ex_decision_boundary.read_bytes() == gen_decision_boundary.read_bytes()

        assert ex_fixed_decision_boundary.read_bytes() == gen_fixed_decision_boundary.read_bytes()

        # clean
        generated_formula.unlink()
        generated_solution.unlink()
        gen_decision_boundary.unlink()
        gen_fixed_decision_boundary.unlink()


if __name__ == '__main__':
    unittest.main()
