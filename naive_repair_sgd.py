"""This module goal is to re-train an unsafe NN with unsat examples w.r.t to some property"""
import logging
import sys
import time
from copy import copy
from datetime import datetime
from pathlib import Path

from sklearn.utils import resample
from z3 import unsat, unknown, sat
import numpy as np

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.exp_config_reader import ExpConfigReader
from nnsynth.common.properties import set_property_from_params
from nnsynth.common.sanity import pred, evaluate_test_acc
from nnsynth.datasets import Dataset, randomly_sample, property_randomly_sample
from nnsynth.evaluate import EvaluateDecisionBoundary, compute_exp_metrics, build_exp_docstring, build_exp_docstring_sgd
from nnsynth.formula_generator import FormulaGenerator
from nnsynth.neural_net import ModularClassificationNet, ClassificationNet, create_skorch_net, get_n_params, \
    get_num_layers, get_params, set_params
from nnsynth.z3_context_manager import Z3ContextManager

CURR_PATH = Path('.')
DATASETS_PATH = CURR_PATH / "datasets"
MODELS_PATH = CURR_PATH / "models"


def main(args):
    logging.info("Init dataset")
    if not args.load_dataset:
        raise Exception("Must pass argument load_dataset")
    else:
        dataset = Dataset.from_pickle(DATASETS_PATH / args.load_dataset)



    X_train_orig, y_train_orig, _, _ = dataset.get_splitted_data()

    logging.info("Init NN instance")
    input_size = dataset.get_input_size()
    num_classes = dataset.get_output_size()
    net_class = ModularClassificationNet if args.modular_nn else ClassificationNet
    net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size, num_classes=num_classes,
                            epochs=args.epochs, learning_rate=args.learning_rate, random_seed=args.random_seed,
                            init=args.load_nn is not None, net_class=net_class)

    # train / load NN
    if args.load_nn:
        net.load_params(MODELS_PATH / args.load_nn)
    else:
        raise Exception("Must pass argument load_nn")

    X_sampled, y_sampled = randomly_sample(net, dataset)
    dataset.add_samples(X_sampled, y_sampled, dataset='sampled')
    dataset_original = copy(dataset)
    original_net = copy(net)
    max_iters = 20

    i = 1
    time_took = 0

    # set exp dir
    _EXP_ROOT_PATH = CURR_PATH / "results-sgd" / f"{Path(args.exp_config_path).stem}"
    _EXP_ROOT_PATH.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%s") if not args.timestamp else args.timestamp
    EXP_PATH = _EXP_ROOT_PATH / f"{timestamp}"
    EXP_PATH.mkdir(exist_ok=True)

    t1 = time.time()
    while True:
        num_layers = get_num_layers(net)
        coefs, intercepts = get_params(net)
        logging.info("Init Formula Generator")
        generator = FormulaGenerator(coefs=coefs, intercepts=intercepts, input_size=input_size,
                                     output_size=num_classes, num_layers=num_layers)
        checked_property = set_property_from_params(properties=args.properties,
                                                    input_size=input_size,
                                                    num_classes=num_classes)

        logging.info("Generate formula")
        generator.generate_formula(checked_property, None, None)

        logging.info("Init Z3 context manager")
        z3_mgr = Z3ContextManager()
        z3_mgr.add_formula_from_memory(generator.get_goal())
        logging.info("Check if SAT...")
        z3_mgr.solve(timeout=args.z3_timeout)
        res = z3_mgr.get_result()

        if res == sat:
            logging.info("SAT formula, stopping.")
            break
        else:
            if i == max_iters:
                logging.info("Reached max iterations, stopping.")
                exit(0)
            else:
                logging.info(f"Result is {res}, continue training.")
                i += 1

        # sample points from properties
        X_prop, y_prop = property_randomly_sample(args.properties, n_samples=args.property_samples_sgd)
        dataset.add_samples(X_prop, y_prop, dataset='train')
        # sample points from original train dataset
        X_train_addition, y_train_addition = resample(X_train_orig, y_train_orig, n_samples=args.property_samples_sgd)
        dataset.add_samples(X_train_addition, y_train_addition)
        # train
        dataset.convert_types_before_train()
        X_train, y_train, X_test, y_test = dataset.get_splitted_data()
        logging.info("Fit...")

        net.fit(X_train, y_train)

    total_time = time.time() - t1

    logging.info("Repaired NN")

    logging.info("Prepare for plot")
    fixed_net = net
    evaluator = EvaluateDecisionBoundary(original_net, fixed_net, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                         contourf_levels=args.contourf_levels, save_plot=args.save_plot,
                                         meshgrid_limit=args.meshgrid_limit)

    metrics = compute_exp_metrics(clf=original_net, fixed_clf=fixed_net, dataset=dataset_original, path=EXP_PATH / "metrics.csv")

    exp_name, details, fname = build_exp_docstring_sgd(args, total_time, get_n_params(net.module),
                                                       get_n_params(net.module), epochs_took=i-1)

    logging.info("Plotting")
    evaluator.multi_plot_all_heuristics(main_details=exp_name, extra_details=details,
                                        keep_ctx_property=None, metrics=metrics, path=EXP_PATH, fname=fname, method="SGD")


if __name__ == '__main__':
    # logger
    logger_format = "%(asctime)s [%(levelname)-5.5s]  %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=logger_format)
    # get args
    args = ArgumentsParser.parser.parse_args()

    logging.info("Loading config")
    CONFIGS_PATH = CURR_PATH / "exp_configs"
    cfg_reader = ExpConfigReader(CONFIGS_PATH / args.exp_config_path)
    cfg_reader.update_args_with_global_config(args)

    logging.info("Starting main")
    main(args)

