import unittest
from filecmp import cmp
from pathlib import Path
from main import main


class TestNNSynthesizer(unittest.TestCase):
    def test_flow(self):
        """Tests output is equal to expected output, when solving for
        specific parameters which for the solution is known"""
        expected_formula = Path('resources/check.smt2')
        # use default arguments
        main()
        generated_formula = Path('check.smt2')

        assert cmp(expected_formula.read_bytes(),
                   generated_formula.read_bytes())

        # clean
        generated_formula.unlink()


if __name__ == '__main__':
    unittest.main()
