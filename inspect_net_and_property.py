"""This module goal is to provide the ability for inspecting the NN and the desired property, it takes
an exp config, similar to the config that `repair_exp_runner.py` expects."""
import logging
import sys
from datetime import datetime
from pathlib import Path


from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.exp_config_reader import ExpConfigReader
from nnsynth.common.properties import set_property_from_params
from nnsynth.datasets import Dataset
from nnsynth.evaluate import EvaluateDecisionBoundary
from nnsynth.neural_net import create_skorch_net, print_params, ModularClassificationNet, ClassificationNet


CURR_PATH = Path('.')
DATASETS_PATH = CURR_PATH / "datasets"
MODELS_PATH = CURR_PATH / "models"


def main(args):
    logging.info("Inspect net and property.")

    logging.info("Init dataset")
    if not args.load_dataset:
        raise Exception("Must pass argument load_dataset")
    else:
        dataset = Dataset.from_pickle(DATASETS_PATH / args.load_dataset)

    logging.info("Init NN instance")
    input_size = dataset.get_input_size()
    num_classes = dataset.get_output_size()
    net_class = ModularClassificationNet if args.modular_nn else ClassificationNet
    net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size, num_classes=num_classes,
                            epochs=args.epochs, learning_rate=args.learning_rate, random_seed=args.random_seed,
                            init=args.load_nn is not None, net_class=net_class)
    if args.load_nn:
        net.load_params(MODELS_PATH / args.load_nn)
        print_params(net)
    else:
        raise Exception("Must pass argument load_nn")

    logging.info("Build property")
    checked_property = set_property_from_params(properties=args.properties,
                                                input_size=input_size,
                                                num_classes=num_classes)

    logging.info("Evaluate property and exit.")
    evaluator = EvaluateDecisionBoundary(net, None, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                         contourf_levels=args.contourf_levels, save_plot=args.save_plot,
                                         meshgrid_limit=args.meshgrid_limit)
    evaluator.plot_with_prop(property=checked_property, path=INSPECT_RESULTS_PATH, name=args.plot_name)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # set CLI args and config args
    args = ArgumentsParser.parser.parse_args()
    CONFIGS_PATH = CURR_PATH / "exp_configs"

    cfg_reader = ExpConfigReader(CONFIGS_PATH / args.exp_config_path)
    cfg_reader.update_args_with_global_config(args)
    key = f"Network: {args.load_nn}, Property: {args.properties}, Dataset: {args.load_dataset}"
    logging.info(key)

    key_parsed = key.replace(' ', '').replace(',', '-')
    INSPECT_RESULTS_PATH = CURR_PATH / "inspector_results" / datetime.utcnow().strftime("%s") / key_parsed
    INSPECT_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    vars(args).update({'plot_name': key_parsed})

    main(args)
