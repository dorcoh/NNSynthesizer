"""This module goal is to provide the ability for serializing the NN."""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path


from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.exp_config_reader import ExpConfigReader
from nnsynth.common.properties import set_property_from_params
from nnsynth.common.sanity import evaluate_dataset
from nnsynth.datasets import Dataset
from nnsynth.evaluate import EvaluateDecisionBoundary
from nnsynth.formula_generator import FormulaGenerator
from nnsynth.neural_net import create_skorch_net, print_params, ModularClassificationNet, ClassificationNet, \
    get_num_layers, get_params, save_params_combs
from nnsynth.z3_context_manager import Z3ContextManager

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
        evaluate_dataset(dataset)

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

    coefs, intercepts = get_params(net)

    weights = {
        'coefs': coefs,
        'intercepts': intercepts,
        'hidden_size': args.hidden_size,
        'properties': args.properties
    }

    json_weights = json.dumps(weights)
    fname = Path(args.load_nn).stem + ".json"
    path_to_save = INSPECT_RESULTS_PATH / fname
    logging.info(f"Saving serialized net to {path_to_save}.")
    with path_to_save.open('w') as handle:
        handle.write(json_weights)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # set CLI args and config args
    args = ArgumentsParser.parser.parse_args()
    CONFIGS_PATH = CURR_PATH / "exp_configs"

    cfg_reader = ExpConfigReader(CONFIGS_PATH / args.exp_config_path)
    cfg_reader.update_args_with_global_config(args)
    _key = f"Network: {args.load_nn}, Property: {args.properties}, Dataset: {args.load_dataset}"
    logging.info(_key)
    key = f"Network: {args.load_nn}"

    config_name_stripped = Path(args.exp_config_path).stem
    INSPECT_RESULTS_PATH = CURR_PATH / "serializer_results" / config_name_stripped / datetime.utcnow().strftime("%s")
    INSPECT_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    main(args)
