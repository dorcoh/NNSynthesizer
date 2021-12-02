"""This module goal is to provide the ability for inspecting the NN and the desired property, it takes
an exp config, similar to the config that `repair_exp_runner.py` expects."""
import logging
import sys
from datetime import datetime
from pathlib import Path

from sklearn.metrics import accuracy_score
import numpy as np

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.exp_config_reader import ExpConfigReader
from nnsynth.common.properties import set_property_from_params
from nnsynth.common.sanity import evaluate_dataset
from nnsynth.datasets import Dataset, randomly_sample
from nnsynth.evaluate import EvaluateDecisionBoundary, compute_exp_metrics
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
        save_params_combs(net, args.load_nn)
    else:
        raise Exception("Must pass argument load_nn")


    logging.info("Add sampled dataset")
    X_sampled, y_sampled = randomly_sample(net, dataset, n_samples=args.sampled_dataset_n)
    dataset.add_samples(X_sampled, y_sampled, dataset='sampled')

    logging.info("Build property")
    checked_property = set_property_from_params(properties=args.properties,
                                                input_size=input_size,
                                                num_classes=num_classes)

    logging.info("Evaluate property")
    evaluator = EvaluateDecisionBoundary(net, None, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                         contourf_levels=args.contourf_levels, save_plot=args.save_plot,
                                         meshgrid_limit=args.meshgrid_limit)
    plot_title = f"Model:{args.plot_name}"
    evaluator.plot_with_prop(property=checked_property, path=INSPECT_RESULTS_PATH, name=plot_title)

    logging.info("Init Formula Generator")
    num_layers = get_num_layers(net)
    coefs, intercepts = get_params(net)
    generator = FormulaGenerator(coefs=coefs, intercepts=intercepts, input_size=input_size,
                                 output_size=num_classes, num_layers=num_layers)

    logging.info("Generating formula in mode: Checking SAT")
    generator.generate_formula(checked_property, None, None)

    z3_mgr = Z3ContextManager()
    z3_mgr.add_formula_from_memory(generator.get_goal())
    z3_mgr.save_formula_to_disk(INSPECT_RESULTS_PATH / 'formula.smt2')
    logging.info("Solve...")
    time_took = z3_mgr.solve(timeout=None)
    logging.info(f"Took: {time_took}")
    res = z3_mgr.get_result()
    logging.info("Check sat mode: formula is {}".format(str(res)))

    logging.info("Compute metrics")
    X_train, y_train, X_test, y_test = dataset.get_splitted_data()

    metrics = {}
    datasets = [('train', X_train, y_train), ('test', X_test, y_test)]
    if hasattr(dataset, 'X_sampled'):
        datasets.append(('sampled', *dataset.get_sampled()))

    for clf_name, _clf in [('original', net)]:
        for set_name, X, y in datasets:
            y_pred = _clf.predict(X)
            n = y.shape[0]
            acc_score = accuracy_score(y, y_pred)
            key = clf_name + f"_{set_name}_acc"
            key_n = key + "_n"
            metrics[key] = acc_score
            metrics[key_n] = n

    orig_avgs = [value for key, value in metrics.items() if key.startswith('original') and not key.endswith('_n')]
    orig_weights = [value for key, value in metrics.items() if key.startswith('original') and key.endswith('_n')]

    metrics['original_avg'] = np.average(orig_avgs, weights=orig_weights)

    logging.info(metrics)

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

    INSPECT_RESULTS_PATH = CURR_PATH / "inspector_results" / datetime.utcnow().strftime("%s") / key
    INSPECT_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    if 'plot_name' not in vars(args):
        vars(args).update({'plot_name': key})

    main(args)
