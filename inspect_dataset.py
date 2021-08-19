"""This module goal is to provide the ability for inspecting the NN and the desired property, it takes
an exp config, similar to the config that `repair_exp_runner.py` expects."""
import logging
import sys
from pathlib import Path


from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.exp_config_reader import ExpConfigReader
from nnsynth.datasets import Dataset


CURR_PATH = Path('.')
DATASETS_PATH = CURR_PATH / "datasets"

def main(args):
    logging.info("Inspect dataset.")

    logging.info("Init dataset")
    if not args.load_dataset:
        raise Exception("Must pass argument load_dataset")
    else:
        dataset = Dataset.from_pickle(DATASETS_PATH / args.load_dataset)

    logging.info("Init NN instance")
    input_size = dataset.get_input_size()
    num_classes = dataset.get_output_size()

    for _set in [('train', dataset.X_train, dataset.y_train), ('test', dataset.X_test, dataset.y_test)]:
        logging.info(f"{_set[0]} dataset: X: {_set[1].shape}")
        logging.info(f"{_set[0]} dataset: y: {_set[2].shape}")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # set CLI args and config args
    args = ArgumentsParser.parser.parse_args()
    CONFIGS_PATH = CURR_PATH / "exp_configs"

    cfg_reader = ExpConfigReader(CONFIGS_PATH / args.exp_config_path)
    cfg_reader.update_args_with_global_config(args)
    key = f"Dataset: {args.load_dataset}"
    logging.info(key)

    main(args)
