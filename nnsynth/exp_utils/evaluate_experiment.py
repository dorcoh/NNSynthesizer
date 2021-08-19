from pathlib import Path

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.utils import load_pickle

args = ArgumentsParser.parser.parse_args()

sub_exp_path = Path('../../experiments/initial/repair-results')
exp_path = sub_exp_path / args.experiment


def deserialize_subexp_results(sub_experiment):
    path = exp_path / sub_experiment
    return load_pickle(path)

for sub_exp in exp_path.iterdir():
    sub_exp_res_dict = deserialize_subexp_results(sub_exp.name)
    res_string = "weight_comb: {}, threshold: {}, distance: {}".format(
        sub_exp_res_dict['weight_comb'],
        sub_exp_res_dict['threshold'],
        sub_exp_res_dict['distance']
    )
    print(res_string)