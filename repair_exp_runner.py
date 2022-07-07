"""
Trigger a set of experiments with configuration file, each instance of experiment takes:
(1) source network and dataset (2) specification, (3) heuristic and its params, it then attempts to perform a repair.
"""
import logging
import sys
import time
from copy import copy
from datetime import datetime
from pathlib import Path

from guppy import hpy
from z3 import unsat, unknown

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.exp_config_reader import ExpConfigReader
from nnsynth.common.properties import EnforceSamplesSoftProperty, EnforceSamplesHardProperty, \
    EnforceGridSoftProperty, KeepContextType, EnforceVoronoiSoftProperty, set_property_from_params
from nnsynth.common.utils import save_exp_config, append_stats, set_stats
from nnsynth.datasets import Dataset, randomly_sample
from nnsynth.evaluate import EvaluateDecisionBoundary, build_exp_docstring, compute_exp_metrics
from nnsynth.formula_generator import FormulaGenerator
from nnsynth.neural_net import create_skorch_net, print_params, get_params, set_params, get_num_layers, \
    ModularClassificationNet, ClassificationNet, get_n_params
from nnsynth.weights_selector import WeightsSelector
from nnsynth.z3_context_manager import Z3ContextManager

CURR_PATH = Path('.')
DATASETS_PATH = CURR_PATH / "datasets"
MODELS_PATH = CURR_PATH / "models"


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

    logging.info("Add sampled dataset")
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
            keep_ctx_property = EnforceSamplesSoftProperty()
            eval_set = dataset.get_evaluate_set(net, args.eval_set, args.eval_set_type, args.limit_eval_set)
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

    logging.info("Generating formula in Repair mode")
    generator.generate_formula(checked_property, weights_selector, keep_ctx_property)

    logging.info("Init Z3 context manager")
    z3_mgr = Z3ContextManager()
    z3_mgr.add_formula_from_memory(generator.get_goal())
    z3_mgr.save_formula_to_disk(EXP_PATH / 'formula.smt2')
    logging.info("Solve...")
    time_took = z3_mgr.solve(timeout=args.z3_timeout)
    res = z3_mgr.get_result()

    # docstring
    exp_name, details, fname = build_exp_docstring(args=args,
                                                   num_constraints=keep_ctx_property.get_num_constraints(),
                                                   time_took=time_took,
                                                   net_params=get_n_params(net.module),
                                                   net_free_params=weights_selector.num_free_weights())

    # exit if not sat
    if (res == unsat or res == unknown) and not args.check_sat:
        logging.info("Stopped with result: " + str(res))
        # cleanup
        del dataset, net
        del X_sampled, y_sampled
        del generator, weights_selector, keep_ctx_property, z3_mgr

        return fname, {'exit-result': str(res)}, time_took

    logging.info("Repaired NN. New weights mapping:")
    model_mapping = z3_mgr.get_model_mapping(generator.get_z3_weight_variables(),
                                             generator.get_original_weight_values())
    logging.info(z3_mgr.model_mapping_sanity_check())

    with open(EXP_PATH / 'model_mapping', 'w') as handle:
        handle.write(str(model_mapping))

    # store original net before fix
    logging.info("Prepare for plot")
    original_net = copy(net)
    fixed_net = set_params(net, model_mapping)
    evaluator = EvaluateDecisionBoundary(original_net, fixed_net, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                         contourf_levels=args.contourf_levels, save_plot=args.save_plot,
                                         meshgrid_limit=args.meshgrid_limit)
    fixed_net.save_params(f_params=EXP_PATH / 'model-fixed.pkl')

    metrics = compute_exp_metrics(clf=original_net, fixed_clf=fixed_net, dataset=dataset, path=EXP_PATH / "metrics.csv")

    logging.info("Plotting")
    evaluator.multi_plot_all_heuristics(main_details=exp_name, extra_details=details,
                                        keep_ctx_property=keep_ctx_property, metrics=metrics, path=EXP_PATH, fname=fname)

    # cleanup
    del dataset, net
    del X_sampled, y_sampled
    del generator, weights_selector, keep_ctx_property, z3_mgr
    del original_net, fixed_net
    del evaluator

    return fname, metrics, time_took


def heap_stats(path, h, exp_id):
    with path.open('a') as handle:
        handle.write(f"-----------------\n")
        handle.write(f"Experiment {exp_id}\n")
        handle.write(str(h.heap()))
        handle.write(f"\n-----------------\n")


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
    timestamp = datetime.utcnow().strftime("%s") if not args.timestamp else args.timestamp
    _EXP_ROOT_PATH = CURR_PATH / "results" / f"{Path(args.exp_config_path).stem}" / timestamp
    _EXP_ROOT_PATH.mkdir(parents=True, exist_ok=True)

    save_exp_config(cfg_reader.get_config_dict(), _EXP_ROOT_PATH / "config.json")
    csv_path = _EXP_ROOT_PATH / "general_stats.csv"

    h = hpy()
    heap_stats(_EXP_ROOT_PATH / "heap.status", h, 0)

    exp_keys = {}
    for i, exp_args_instance in cfg_reader.get_experiments_instances(args, csv_path):
        # exp instance path
        EXP_PATH = _EXP_ROOT_PATH / f"exp_{i + 1}"
        EXP_PATH.mkdir(exist_ok=True)

        # logger
        logFormatter = logging.Formatter(logger_format)
        rootLogger = logging.getLogger()
        fileHandler = logging.FileHandler(EXP_PATH / "log.out")
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        vars(args).update(exp_args_instance)
        if exp_keys.get(key_without_threshold(args)):
            # assuming thresholds are increasing, we can earn from 'early stopping'
            logging.info(f"Skipping exp {i + 1}, similar configs with lower thresholds resulted with UNSAT/UNKNOWN")
            skipped_exp_key = skipped_exp_id(args)
            logging.info(f"Exp id: {skipped_exp_key}")
            set_stats(path=_EXP_ROOT_PATH / "general_stats.csv", exp_id=i + 1, exp_key=skipped_exp_key,
                         metrics={'exit-result': 'skipped'},
                         time_took='skipped', extra=None)

            # save args
            save_exp_config(vars(args), EXP_PATH / "config.json")
            # clear logger
            logging.info(f"Clear logger")
            rootLogger.removeHandler(fileHandler)
            heap_stats(_EXP_ROOT_PATH / "heap.status", h, exp_id=i + 1)
            continue

        # actual run
        logging.info(f"Starting exp {i + 1}")

        start = time.time()
        fname, metrics, time_took = main(args)

        if metrics.get('exit-result'):
            id = key_without_threshold(args)
            exp_keys[id] = True

        extra_metadata = str(args.heuristic_params) if args.heuristic_params else None
        set_stats(path=_EXP_ROOT_PATH / "general_stats.csv", exp_id=i + 1, exp_key=fname, metrics=metrics,
                     time_took=time_took, extra=extra_metadata)
        logging.info(f"Total time: {time.time() - start} seconds.")

        # save args
        save_exp_config(vars(args), EXP_PATH / "config.json")
        # clear logger
        logging.info(f"Clear logger")
        rootLogger.removeHandler(fileHandler)
        heap_stats(_EXP_ROOT_PATH / "heap.status", h, exp_id=i+1)