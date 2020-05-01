"""Main module for (1) generating the SMT formula (2) solve the formula"""
import pickle

from z3 import unsat, unknown

from nnsynth.common import sanity
from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.models import OutputConstraint
from nnsynth.common.properties import KeepContextProperty, DeltaRobustnessProperty
from nnsynth.common.sanity import xor_dataset_sanity_check
from nnsynth.datasets import XorDataset
from nnsynth.formula_generator import FormulaGenerator
from nnsynth.neural_net import get_params, create_skorch_net, print_params, get_num_layers
from nnsynth.weights_selector import WeightsSelector
from nnsynth.z3_context_manager import Z3ContextManager


def main(args):
    # main flow

    # load dataset
    if args.load_dataset:
        dataset = XorDataset.from_pickle(args.load_dataset)
    else:
        exit(1)

    input_size = dataset.get_input_size()
    num_classes = dataset.get_output_size()

    net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size,
                            num_classes=num_classes, learning_rate=args.learning_rate,
                            epochs=args.epochs, random_seed=args.random_seed,
                            init=args.load_nn is not None)
    # train / load NN
    if args.load_nn:
        net.load_params(args.load_nn)
    else:
        exit(1)

    print_params(net)

    num_layers = get_num_layers(net)

    # formulate in SMT via z3py
    coefs, intercepts = get_params(net)
    generator = FormulaGenerator(coefs=coefs, intercepts=intercepts, input_size=input_size,
                                 output_size=num_classes, num_layers=num_layers)
    checked_property = [
        DeltaRobustnessProperty(input_size=input_size, output_size=num_classes, desired_output=1,
                                coordinate=args.pr_coordinate, delta=args.pr_delta,
                                output_constraint_type=OutputConstraint.Max)
        ]

    # TODO: wrap weights selector in some tactic generator or heuristic search,
    #  configure robustness property and weights selection
    # TODO: change hidden size type
    weights_selector = WeightsSelector(input_size=input_size, hidden_size=(4,),
                                       output_size=num_classes, delta=args.ws_delta)

    # keep context (original NN representation)
    eval_set = dataset.get_evaluate_set(net, args.eval_set, args.eval_set_type, 50)
    sanity.print_eval_set(eval_set)
    eval_set = XorDataset.filter_eval_set(eval_set)
    keep_ctx_property = KeepContextProperty(eval_set)

    # all combinations for 2-4-2 NN
    combinations = {
        # layer, neuron, weight
        'weights': [
            (1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (1, 3, 1), (1, 3, 2), (1, 4, 1), (1, 4, 2),
            (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2)
        # layer, neuron
        ],
        'biases': [
            (1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2)
        ]
    }
    # all thresholds
    thresholds = reversed([i for i in range(10)])

    # TODO: add the biases as well to the search
    for weight in combinations['weights']:
        weights_selector.select_weight(weight[0], weight[1], weight[2])

        for threshold in thresholds:
            keep_ctx_property.set_threshold(threshold)

            generator.generate_formula(checked_property, weights_selector, keep_ctx_property)

            z3_mgr = Z3ContextManager('check.smt2')
            z3_mgr.add_formula_to_z3_memory(generator.get_goal())

            # TODO: this call should invoke PBS job
            #send_pbs_job(...)

    # TODO: the PBS job runs the code below

    z3_mgr = Z3ContextManager()
    # TODO: add formula name/path as paramter for this script
    z3_mgr.add_formula_to_z3_disk('formula-path-should-get-as-parameter')

    z3_mgr.solve()

    res = z3_mgr.get_result()

    # exit if not sat
    if (res == unsat or res == unknown):
        print("Stopped with result: " + str(res))
        exit(1)

    model_mapping = z3_mgr.get_model_mapping(generator.get_z3_weight_variables(),
                                             generator.get_original_weight_values())

    z3_mgr.model_mapping_sanity_check()

    # TODO - assign the file name with the formula name (add some key as identifier)
    with open('model_mapping', 'wb') as handle:
        pickle.dump(model_mapping, handle, pickle.HIGHEST_PROTOCOL)

    print(xor_dataset_sanity_check(net))

    # TODO: think of a way to organize the results (including configurations)

    # TODO: mark the current instance as sat/unsat/unknown on some csv (append results for each formula)
    #  should list the configurations as well


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)
