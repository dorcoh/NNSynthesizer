"""Solve an instance of main loop"""
import sys
from pathlib import Path

from z3 import unsat, unknown

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.models import OutputConstraint
from nnsynth.common.properties import KeepContextProperty, DeltaRobustnessProperty
from nnsynth.common.utils import deserialize_exp, save_exp_details, save_pickle, load_pickle
from nnsynth.formula_generator import FormulaGenerator
from nnsynth.weights_selector import WeightsSelector
from nnsynth.z3_context_manager import Z3ContextManager


def main(args):
    # main flow

    exp = deserialize_exp(args.exp_filename)
    sub_exp = load_pickle(args.sub_exp_filename)

    generator = FormulaGenerator(coefs=exp['coefs'], intercepts=exp['intercepts'],
                                 input_size=exp['input_size'],
                                 output_size=exp['num_classes'], num_layers=exp['num_layers'])

    # 1 delta robustness property at center (10, 10)
    checked_property = [
        DeltaRobustnessProperty(input_size=exp['input_size'], output_size=exp['num_classes'],
                                desired_output=1, coordinate=args.pr_coordinate, delta=args.pr_delta,
                                output_constraint_type=OutputConstraint.Max)
        ]

    weights_selector = WeightsSelector(input_size=exp['input_size'], hidden_size=(4,),
                                       output_size=exp['num_classes'], delta=args.ws_delta)

    # keep context (original NN representation)
    keep_ctx_property = KeepContextProperty(exp['eval_set'])

    # debug

    threshold = sub_exp['threshold']
    eval_set_size = sub_exp['eval_set_size']
    weight_tuple = sub_exp['weight_comb']

    weights_selector.reset_selected_weights()
    # weights
    if weight_tuple[0] == 'w':
        weights_selector.select_weight(weight_tuple[1], weight_tuple[2], weight_tuple[3])
    # biases
    elif weight_tuple[0] == 'b':
        weights_selector.select_bias(weight_tuple[1], weight_tuple[2])

    model_config = {
        'weight_comb': weight_tuple,
        'threshold': threshold,
        'eval_set_size': eval_set_size
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

    # exit if not sat
    if (res == unsat or res == unknown) and not args.check_sat:
        print("Stopped iteration with result: " + str(res))
        save_exp_details(model_config, res, None, None,
                         args.sub_exp_filename.split('.pkl')[0] + '.results')
        sys.exit(1)

    # sat - save model, model mapping and the exp details
    else:

        model_mapping = z3_mgr.get_model_mapping(generator.get_z3_weight_variables(),
                                                 generator.get_original_weight_values())
        # TODO: for multiple weights we should use average
        param_dist = abs(model_mapping[weights_selector.selected_weights[0]][0] -
                   model_mapping[weights_selector.selected_weights[0]][1])

        z3_mgr.model_mapping_sanity_check()
        save_exp_details(model_config, res, param_dist, model_mapping,
                         args.sub_exp_filename.split('.pkl')[0] + '.results')


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)
