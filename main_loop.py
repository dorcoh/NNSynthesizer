"""Main loop for repairing an NN - each iteration runs on the same process"""
from pathlib import Path

from z3 import unsat, unknown

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.models import OutputConstraint
from nnsynth.common.properties import KeepContextProperty, DeltaRobustnessProperty
from nnsynth.common.utils import deserialize_exp, save_exp_details, save_pickle
from nnsynth.formula_generator import FormulaGenerator
from nnsynth.weights_selector import WeightsSelector
from nnsynth.z3_context_manager import Z3ContextManager


def main(args):
    # main flow

    exp = deserialize_exp()

    generator = FormulaGenerator(coefs=exp['coefs'], intercepts=exp['intercepts'],
                                 input_size=exp['input_size'],
                                 output_size=exp['num_classes'], num_layers=exp['num_layers'])
    checked_property = [
        DeltaRobustnessProperty(input_size=exp['input_size'], output_size=exp['num_classes'],
                                desired_output=1, coordinate=args.pr_coordinate, delta=args.pr_delta,
                                output_constraint_type=OutputConstraint.Max)
        ]

    weights_selector = WeightsSelector(input_size=exp['input_size'], hidden_size=(4,),
                                       output_size=exp['num_classes'], delta=args.ws_delta)

    # keep context (original NN representation)
    keep_ctx_property = KeepContextProperty(exp['eval_set'])

    # all combinations for 2-4-2 NN
    _comb_tuples = [
        # weights: layer, neuron, weight
        # bias: layer, neuron
        ('w', 1, 1, 1), ('w', 1, 1, 2), ('b', 1, 1),
        ('w', 1, 2, 1), ('w', 1, 2, 2), ('b', 1, 2),
        ('w', 1, 3, 1), ('w', 1, 3, 2), ('b', 1, 3),
        ('w', 1, 4, 1), ('w', 1, 4, 2), ('b', 1, 4),
        ('w', 2, 1, 1), ('w', 2, 1, 2), ('b', 2, 1),
        ('w', 2, 2, 1), ('w', 2, 2, 2), ('b', 2, 2),
    ]

    # all thresholds
    thresholds = list(reversed([i for i in range(1, args.limit_eval_set+1)]))

    # currently we measure this by the distance abs(original_weight_value - searched_weight_value)
    best_model_distance = 1e5
    best_model_config = ''
    best_model_mapping = {}

    # debug

    for weight_tuple in _comb_tuples[:9]:
        weights_selector.reset_selected_weights()
        # weights
        if weight_tuple[0] == 'w':
            weights_selector.select_weight(weight_tuple[1], weight_tuple[2], weight_tuple[3])
        # biases
        elif weight_tuple[0] == 'b':
            weights_selector.select_bias(weight_tuple[1], weight_tuple[2])

        for threshold in thresholds[:1]:
            # TODO: add invocation of single jobs to Tamnun
            model_config = {
                'weight_comb': weight_tuple,
                'threshold': threshold,
                'eval_set_size': exp['eval_set'][0].shape[0]
            }
            print("Iteration details: %s" % str(model_config))

            keep_ctx_property.set_threshold(threshold)

            generator.reset_weight_variables_values()
            generator.generate_formula(checked_property, weights_selector, keep_ctx_property)

            z3_mgr = Z3ContextManager()
            z3_mgr.add_formula_from_memory(generator.get_goal())

            # invoke the solver from python wrapper
            z3_mgr.solve()

            res = z3_mgr.get_result()

            # continue to next experiment if not sat
            if (res == unsat or res == unknown) and not args.check_sat:
                print("Stopped iteration with result: " + str(res))
                save_exp_details(model_config, res, None)
                continue

            else:
                # sat - save model and break thresholds loop (found the maximal threshold -
                # continue the search with another weight combination)
                model_mapping = z3_mgr.get_model_mapping(generator.get_z3_weight_variables(),
                                                         generator.get_original_weight_values())
                # TODO: for multiple weights we should use average
                param_dist = abs(model_mapping[weights_selector.selected_weights[0]][0] -
                           model_mapping[weights_selector.selected_weights[0]][1])

                # TODO: remove if we want to send single jobs to Tamnun
                if args.send_single and param_dist < best_model_distance:
                    best_model_distance = param_dist
                    # TODO: change the way we measure models performance (use test set)
                    best_model_mapping = model_mapping
                    best_model_config = model_config

                z3_mgr.model_mapping_sanity_check()
                save_exp_details(model_config, res, param_dist)


    # save best model results as dictionary
    best_model_results = best_model_config

    best_model_results_path = Path('best_model_results')
    save_pickle(best_model_results, best_model_results_path)

    best_model_mapping_path = Path('best_model_mapping')
    save_pickle(best_model_mapping, best_model_mapping_path)


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)
