"""
Pre-compute config files for an experiments set, takes a config and generates config files in `cache` directory. The
goal of caching is for a quick mitigation in case of failure (can restart from the last experiment).
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.exp_config_reader import ExpConfigReader
from nnsynth.common.properties import KeepContextType
from nnsynth.common.utils import save_exp_config, append_stats

CURR_PATH = Path('.')
DATASETS_PATH = CURR_PATH / "datasets"
MODELS_PATH = CURR_PATH / "models"


def skipped_exp_id(args):
    id = f"RepairResult::Props-{args.num_properties}::Heuristic-{KeepContextType(args.heuristic).name}::" \
            f"NumConstraints-{None}::Threshold-{args.threshold}::NNHidden-{args.hidden_size}::" \
            f"Params-{None}::Free-{None}::WeightsConfig-{args.weights_config}"
    return id

def key_without_threshold(args):
    id = f"RepairResult::Props-{args.num_properties}::Heuristic-{KeepContextType(args.heuristic).name}::" \
            f"NumConstraints-{None}::Threshold-{None}::NNHidden-{args.hidden_size}::" \
            f"Params-{None}::Free-{None}::WeightsConfig-{args.weights_config}"
    return id


if __name__ == '__main__':
    logger_format = "%(asctime)s [%(levelname)-5.5s]  %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=logger_format)
    # set CLI args and config args
    args = ArgumentsParser.parser.parse_args()
    CONFIGS_PATH = CURR_PATH / "exp_configs"
    cfg_reader = ExpConfigReader(CONFIGS_PATH / args.exp_config_path)
    cfg_reader.update_args_with_global_config(args)

    # set exp dir
    _EXP_ROOT_PATH = CURR_PATH / "cache" / f"{Path(args.exp_config_path).stem}"
    _EXP_ROOT_PATH.mkdir(parents=True, exist_ok=True)
    save_exp_config(cfg_reader.get_config_dict(), _EXP_ROOT_PATH / "config.json")

    for i, exp_args_instance in cfg_reader.get_experiments_instances(args, None):
        logging.info(f"Exp {i + 1}")
        # exp instance path
        EXP_PATH = _EXP_ROOT_PATH / f"exp_{i + 1}"
        EXP_PATH.mkdir(exist_ok=True)
        config_path = EXP_PATH / "config.json"

        vars(args).update(exp_args_instance)
        skipped_exp_key = skipped_exp_id(args)
        logging.info(f"Exp key: {skipped_exp_key}")
        append_stats(path=_EXP_ROOT_PATH / "general_stats_cached.csv", exp_id=i + 1, exp_key=None,
                     metrics={},
                     time_took=None, extra=None, config_path=config_path.absolute())
        # save args
        save_exp_config(vars(args), config_path)
