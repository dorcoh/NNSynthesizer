import unittest
from pathlib import Path

from nnsynth.common.utils import load_pickle
from nnsynth.evaluate import EvaluateDecisionBoundary


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        patches_patch = Path('resources/test_patches.pkl')
        self.patches = load_pickle(patches_patch)

    def test_plot_patches(self):
        EvaluateDecisionBoundary.plot_patches(*self.patches)


if __name__ == '__main__':
    unittest.main()
