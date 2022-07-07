"""Single experiement serializer, can choose:
1. Dataset
2. Neural Net architecture
3. Evaluate set
"""

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.common.models import OutputConstraint
from nnsynth.common.properties import DeltaRobustnessProperty
from nnsynth.common.utils import serialize_exp
from nnsynth.datasets import XorDataset, Dataset
from nnsynth.neural_net import get_num_layers, print_params, create_skorch_net, get_params, ClassificationNet, \
    ModularClassificationNet


def main(args):
    # main flow

    # load dataset
    if args.load_dataset:
        dataset = Dataset.from_pickle(args.load_dataset)
    else:
        exit(1)

    input_size = dataset.get_input_size()
    num_classes = dataset.get_output_size()

    # dataset.filter_data(args.eval_set, dataset.is_noisy_sample)
    # # notice it is only 'outside' property (if property enforced first quarter)
    # dataset.filter_data(args.eval_set, dataset.in_first_quarter)

    net_class = ModularClassificationNet if args.modular_nn else ClassificationNet
    net = create_skorch_net(input_size=input_size, hidden_size=args.hidden_size, num_classes=num_classes,
                            epochs=args.epochs, learning_rate=args.learning_rate, random_seed=args.random_seed,
                            init=args.load_nn is not None, net_class=net_class)
    if args.load_nn:
        net.load_params(args.load_nn)
    else:
        exit(1)

    print_params(net)

    property_list = [
        {
            'class_name': 'DeltaRobustnessProperty',
            'input_size': input_size,
            'output_size': num_classes,
            'desired_output': 1,
            'coordinate': args.pr_coordinate,
            'delta': args.pr_delta,
            'output_constraint_type': OutputConstraint.Max
        }
    ]

    num_layers = get_num_layers(net)
    coefs, intercepts = get_params(net)

    eval_set = dataset.get_evaluate_set(net, args.eval_set, args.eval_set_type, args.limit_eval_set)
    serialize_exp(input_size, num_classes, num_layers, coefs, intercepts, eval_set, args.experiment, args.hidden_size,
                  property_list)


if __name__ == '__main__':
    # get args
    args = ArgumentsParser.parser.parse_args()
    main(args)
