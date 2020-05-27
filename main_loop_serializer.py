"""Serializing instances of the main loop"""
from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.models import OutputConstraint
from nnsynth.common.properties import KeepContextProperty, DeltaRobustnessProperty
from nnsynth.common.utils import deserialize_exp, serialize_main_loop_instance
from nnsynth.formula_generator import FormulaGenerator
from nnsynth.weights_selector import WeightsSelector


def generate_main_loop_instance_filename(experiment, weight_tuple, threshold, eval_set_size):
    filename = "exp__" + experiment
    filename += "__subexp__evalsize_" + str(eval_set_size)
    filename += "_threshold_" + str(threshold)
    filename += "_param"
    for elem in weight_tuple:
        filename += "_" + str(elem)
    filename += ".pkl"
    return filename


def main(args):
    # main flow
    exp = deserialize_exp(args.experiment)

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

    # debug
    i = 0
    for weight_tuple in _comb_tuples:
        for threshold in thresholds:
            filename = generate_main_loop_instance_filename(args.experiment, weight_tuple, threshold, exp['eval_set'][0].shape[0])
            serialize_main_loop_instance(weight_tuple, threshold, exp['eval_set'][0].shape[0], filename)
            i += 1
            if i == 1:
                import sys; sys.exit(0)


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)
