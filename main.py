"""Demonstrate the flow by invoking a repair for single network"""
import json
import logging
import sys
import time
from copy import copy
from datetime import datetime
from pathlib import Path

from z3 import unsat, unknown

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.properties import EnforceSamplesSoftProperty, EnforceSamplesHardProperty, \
    EnforceGridSoftProperty, KeepContextType, EnforceVoronoiSoftProperty, set_property_from_params
from nnsynth.common.sanity import xor_dataset_sanity_check
from nnsynth.common.utils import save_exp_config
from nnsynth.datasets import Dataset, randomly_sample
from nnsynth.evaluate import EvaluateDecisionBoundary, build_exp_docstring, compute_exp_metrics
from nnsynth.formula_generator import FormulaGenerator
from nnsynth.neural_net import create_skorch_net, print_params, get_params, set_params, get_num_layers, \
    ModularClassificationNet, ClassificationNet, get_n_params
from nnsynth.weights_selector import WeightsSelector
from nnsynth.z3_context_manager import Z3ContextManager

CURR_PATH = Path('.')
TMP_PATH = CURR_PATH / "results-main" / datetime.utcnow().strftime("%s")
DATASETS_PATH = CURR_PATH / "datasets"
MODELS_PATH = CURR_PATH / "models"
TMP_PATH.mkdir(exist_ok=True, parents=True)


def main(args):
    logging.info("Starting main flow.")

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

    X_sampled, y_sampled = randomly_sample(net, dataset, n_samples=args.sampled_dataset_n)
    dataset.add_samples(X_sampled, y_sampled, dataset='sampled')

    logging.info("Init Formula Generator")
    num_layers = get_num_layers(net)
    coefs, intercepts = get_params(net)
    generator = FormulaGenerator(coefs=coefs, intercepts=intercepts, input_size=input_size,
                                 output_size=num_classes, num_layers=num_layers)

    logging.info("Build property")
    checked_property = set_property_from_params(properties=args.properties,
                                                input_size=input_size,
                                                num_classes=num_classes)

    logging.info("Select free weights")
    weights_selector = WeightsSelector(input_size=input_size, hidden_size=args.hidden_size,
                                       output_size=num_classes, delta=args.ws_delta)
    weights_selector.auto_select(args.weights_config)

    logging.info("Init Similarity Heuristic")
    if args.soft_constraints:
        logging.info("Soft constraints with heuristic {} were activated".format(args.heuristic))
        if args.heuristic == KeepContextType.SAMPLES.value:
            eval_set = dataset.get_evaluate_set(net, args.eval_set, args.eval_set_type, args.limit_eval_set)
            keep_ctx_property = EnforceSamplesSoftProperty()
            keep_ctx_property.set_kwargs(**{'eval_set': eval_set, 'threshold': args.threshold})
        elif args.heuristic == KeepContextType.GRID.value:
            keep_ctx_property = EnforceGridSoftProperty(net=net, **args.heuristic_params)
            keep_ctx_property.set_kwargs(**{'threshold': args.threshold})
        elif args.heuristic == KeepContextType.VORONOI.value:
            keep_ctx_property = EnforceVoronoiSoftProperty(**args.heuristic_params)
            keep_ctx_property.set_kwargs(**{'eval_set': dataset.get_sampled(), 'threshold': args.threshold})
    else:
        logging.info("Hard constraints with heuristic 1 were activated")
        keep_ctx_property = EnforceSamplesHardProperty()

    if args.check_sat:
        logging.info("Generating formula in mode: Checking SAT")
        generator.generate_formula(checked_property, None, None)
    else:
        logging.info("Generating formula in mode: Repair")
        generator.generate_formula(checked_property, weights_selector, keep_ctx_property)

    z3_mgr = Z3ContextManager()
    z3_mgr.add_formula_from_memory(generator.get_goal())
    z3_mgr.save_formula_to_disk(TMP_PATH / 'formula.smt2')
    logging.info("Solve...")
    time_took = z3_mgr.solve(timeout=args.z3_timeout)
    res = z3_mgr.get_result()

    # exit if not sat
    if (res == unsat or res == unknown) and not args.check_sat:
        logging.info("Stopped with result: " + str(res))
        return 1

    elif args.check_sat:
        # check sat mode logic (no weights are freed, or additional constraints added)
        logging.info("Check sat mode: formula is {}".format(str(res)))
        exit(0)

    logging.info("Repaired NN. New weights mapping:")
    model_mapping = z3_mgr.get_model_mapping(generator.get_z3_weight_variables(),
                                             generator.get_original_weight_values())
    logging.info(z3_mgr.model_mapping_sanity_check())

    with open(TMP_PATH / 'model_mapping', 'w') as handle:
        handle.write(str(model_mapping))

    # store original net before fix
    original_net = copy(net)

    logging.info("Prepare for plot")
    fixed_net = set_params(net, model_mapping)
    evaluator = EvaluateDecisionBoundary(original_net, fixed_net, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                         contourf_levels=args.contourf_levels, save_plot=args.save_plot)
    fixed_net.save_params(f_params=TMP_PATH / 'model-fixed.pkl')

    metrics = compute_exp_metrics(clf=original_net, fixed_clf=fixed_net, dataset=dataset, path=TMP_PATH / "metrics.csv")

    # docstring for the plot
    exp_name, details, fname = build_exp_docstring(args=args, num_constraints=keep_ctx_property.get_num_constraints(),
                                                   time_took=time_took,
                                                   net_params=get_n_params(net.module),
                                                   net_free_params=weights_selector.num_free_weights())

    logging.info("Plotting")
    evaluator.multi_plot_all_heuristics(main_details=exp_name, extra_details=details,
                                        keep_ctx_property=keep_ctx_property, metrics=metrics, path=TMP_PATH,
                                        fname=fname)

    logging.info("Sanity checks")
    logging.info(xor_dataset_sanity_check(original_net))
    logging.info(xor_dataset_sanity_check(fixed_net))


if __name__ == '__main__':
    # get args
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(TMP_PATH / "log.out")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    args = ArgumentsParser.parser.parse_args()

    # patch args with config
    if args.config is not None:
        with Path(args.config).open('r') as handle:
            _args = json.load(handle)
            vars(args).update(_args)
    else:
        raise Exception("Must supply config")

    start = time.time()
    main(args)
    logging.info(f"Total time: {time.time() - start} seconds.")
    save_exp_config(vars(args), TMP_PATH / "config.json")
