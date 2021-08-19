"""Demonstrate the flow by invoking a repair for single network"""
import logging
import sys
import time
from copy import copy
from datetime import datetime
from pathlib import Path

from z3 import unsat, unknown

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.models import OutputConstraint
from nnsynth.common.properties import DeltaRobustnessProperty, EnforceSamplesSoftProperty, EnforceSamplesHardProperty, \
    EnforceGridSoftProperty, KeepContextType, EnforceVoronoiSoftProperty
from nnsynth.common.sanity import xor_dataset_sanity_check, evaluate_test_acc, pred
from nnsynth.common.utils import save_exp_config
from nnsynth.datasets import Dataset, randomly_sample
from nnsynth.evaluate import EvaluateDecisionBoundary, build_exp_docstring
from nnsynth.formula_generator import FormulaGenerator
from nnsynth.neural_net import create_skorch_net, print_params, get_params, set_params, get_num_layers, \
    ModularClassificationNet, ClassificationNet, get_n_params
from nnsynth.weights_selector import WeightsSelector
from nnsynth.z3_context_manager import Z3ContextManager

CURR_PATH = Path('.')
TMP_PATH = CURR_PATH / "tmp" / datetime.utcnow().strftime("%s")
DATASETS_PATH = CURR_PATH / "datasets"
MODELS_PATH = CURR_PATH / "models"
TMP_PATH.mkdir(exist_ok=True)


def main(args):
    # main flow
    # generate data and split
    if not args.load_dataset:
        raise Exception("Must pass argument load_dataset")
    else:
        dataset = Dataset.from_pickle(DATASETS_PATH / args.load_dataset)

    X_train, y_train, X_test, y_test = dataset.get_splitted_data()

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
        # net.fit(X_train, y_train)

    if args.eval_nn_and_exit:
        evaluator = EvaluateDecisionBoundary(net, None, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                             contourf_levels=args.contourf_levels, save_plot=False,
                                             meshgrid_limit=args.meshgrid_limit)
        evaluator.plot(use_test=False)
        exit(0)

    X_sampled, y_sampled = randomly_sample(net, dataset, n_samples=200)
    dataset.add_samples(X_sampled, y_sampled, dataset='sampled')

    print_params(net)

    num_layers = get_num_layers(net)

    z3_mgr = Z3ContextManager()

    logging.info("Init Formula Generator")
    coefs, intercepts = get_params(net)
    generator = FormulaGenerator(coefs=coefs, intercepts=intercepts, input_size=input_size,
                                 output_size=num_classes, num_layers=num_layers)

    logging.info("Build property")
    checked_property = [
        DeltaRobustnessProperty(input_size=input_size, output_size=num_classes, desired_output=args.pr_desired_output,
                                coordinate=args.pr_coordinate, delta=args.pr_delta,
                                output_constraint_type=OutputConstraint.Max)
        ]

    if args.eval_nn_and_property_and_exit:
        logging.info("Evaluate property and exit.")
        evaluator = EvaluateDecisionBoundary(net, None, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                             contourf_levels=args.contourf_levels, save_plot=False,
                                             meshgrid_limit=args.meshgrid_limit)
        evaluator.plot_with_prop(property=checked_property)
        exit(0)

    logging.info("Select free weights")
    weights_selector = WeightsSelector(input_size=input_size, hidden_size=args.hidden_size,
                                       output_size=num_classes, delta=args.ws_delta)
    # weights_selector.auto_select(args.weights_config_path)
    weights_selector.select_neuron(layer=2, neuron=1)
    # weights_selector.select_neuron(layer=2, neuron=2)

    # TODO: remove? get actual eval_set
    if not args.dev:
        eval_set = dataset.get_evaluate_set(net, args.eval_set, args.eval_set_type, args.limit_eval_set)
    else:
        eval_set = dataset.get_dummy_eval_set(args.num_properties)

    # sanity.print_eval_set(eval_set)
    logging.debug("Eval set and its predictions")
    logging.debug(eval_set)
    logging.debug(pred(net, eval_set[0]))

    if args.soft_constraints:
        logging.info("Soft constraints with heuristic {} were activated".format(args.heuristic))
        if args.heuristic == KeepContextType.SAMPLES.value:
            keep_ctx_property = EnforceSamplesSoftProperty()
            keep_ctx_property.set_kwargs(**{'eval_set': eval_set, 'threshold': args.threshold})
        elif args.heuristic == KeepContextType.GRID.value:
            keep_ctx_property = EnforceGridSoftProperty(net, x_range=(-22, 22), y_range=(-22, 22), grid_delta=2, samples_num=3, limit_cells=500)
            keep_ctx_property.set_kwargs(**{'threshold': args.threshold})
        elif args.heuristic == KeepContextType.VORONOI.value:
            keep_ctx_property = EnforceVoronoiSoftProperty()
            keep_ctx_property.set_kwargs(**{'eval_set': dataset.get_sampled(), 'threshold': args.threshold})
    else:
        keep_ctx_property = EnforceSamplesHardProperty()

    if args.check_sat:
        logging.info("Generating formula in mode: Checking SAT")
        generator.generate_formula(checked_property, None, None)
    else:
        logging.info("Generating formula in mode: Repair")
        generator.generate_formula(checked_property, weights_selector, keep_ctx_property)
        # generator.generate_formula(checked_property, weights_selector, None)  # no keep ctx

    z3_mgr.add_formula_from_memory(generator.get_goal())

    #keep_ctx_property.save_patches('keep_context_voronoi_patches_xor-bad.pickle')

    z3_mgr.save_formula_to_disk(TMP_PATH / 'formula-{}-xor-bad.smt2'.format(keep_ctx_property.get_constraints_type()))
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

    logging.info("New weights mapping:")
    model_mapping = z3_mgr.get_model_mapping(generator.get_z3_weight_variables(),
                                             generator.get_original_weight_values())
    # debug (for setting results of z3 solver) - can set here your params
    # model_mapping = OrderedDict([('weight_1_1_1', (0.5993294617618902, 0.6119044423103333))])
    # z3_mgr.set_model_mapping(model_mapping)
    logging.info(z3_mgr.model_mapping_sanity_check())

    with open(TMP_PATH / 'main.py-model_mapping', 'w') as handle:
        handle.write(str(model_mapping))

    # store original net before fix
    original_net = copy(net)

    # add eval set to train
    # dataset.add_samples(eval_set[0], eval_set[1], dataset='train')
    logging.info("Prepare for plot")

    fixed_net = set_params(net, model_mapping)
    evaluator = EvaluateDecisionBoundary(original_net, fixed_net, dataset, meshgrid_stepsize=args.meshgrid_stepsize,
                                         contourf_levels=args.contourf_levels, save_plot=args.save_plot)
    fixed_net.save_params(f_params=TMP_PATH / 'model-fixed-voronoi-xor-bad.pkl')

    # docstring for the plot
    exp_name, details, fname = build_exp_docstring(args=args, num_constraints=keep_ctx_property.get_num_constraints(),
                                            time_took=time_took,
                                            net_params=get_n_params(net.module),
                                            net_free_params=weights_selector.num_free_weights())

    logging.info("Plotting")
    if args.heuristic == KeepContextType.SAMPLES.value:
        evaluator.multi_plot(eval_set, name=exp_name, sub_name=details, fname=TMP_PATH / fname)
    elif args.heuristic == KeepContextType.GRID.value or args.heuristic == KeepContextType.VORONOI.value:
        patches, patches_labels = keep_ctx_property.get_patches()
        # evaluator.plot_patches(patches, patches_labels)
        evaluator.multi_plot_with_patches(patches=patches, patches_labels=patches_labels, exp_name=exp_name,
                                          sub_name=details, name=TMP_PATH / fname)

    logging.info("Sanity checks")
    logging.info(xor_dataset_sanity_check(original_net))
    logging.info(xor_dataset_sanity_check(fixed_net))

    # TODO: put these inside evaluator methods
    evaluate_test_acc(original_net, X_test, y_test)
    evaluate_test_acc(fixed_net, X_test, y_test)


if __name__ == '__main__':
    # get args
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(TMP_PATH / "log.out")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    args = ArgumentsParser.parser.parse_args()
    start = time.time()
    main(args)
    logging.info(f"Total time: {time.time() - start} seconds.")
    save_exp_config(vars(args), TMP_PATH / "config.json")
