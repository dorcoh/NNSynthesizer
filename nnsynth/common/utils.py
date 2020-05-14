import os
import pickle
from pathlib import Path
from typing import Dict


def serialize_exp(input_size, num_classes, num_layers, coefs, intercepts, eval_set, filename='ser.exp'):
    serialization_dict = {
        'experiment': "Some-identifers",
        'input_size': input_size,
        'num_classes': num_classes,
        'num_layers': num_layers,
        'coefs': coefs,
        'intercepts': intercepts,
        'eval_set': eval_set
    }

    with open(filename, 'wb') as handle:
        pickle.dump(serialization_dict, handle, pickle.HIGHEST_PROTOCOL)


def deserialize_exp(filename='ser.exp'):
    return load_pickle(filename)


def parse_weight_comb(weight_comb: tuple):
    """Parse weight combination tuple to string"""
    s = ""
    for i, elem in enumerate(weight_comb):
        s += str(elem)
        if i != len(weight_comb)-1:
            s += '-'
    return s


def save_exp_details(model_config: Dict, result, distance):
    results_path = Path('repair-results')
    if not results_path.exists():
        results_path.mkdir()

    model_config['result'] = result
    model_config['distance'] = distance
    results_file_name = 'result_dict_' + parse_weight_comb(model_config['weight_comb']) + '_t-' + str(model_config['threshold'])

    save_pickle(model_config, results_path / results_file_name)


def load_pickle(filename):
    if isinstance(filename, str):
        with open(filename, 'rb') as handle:
            obj = pickle.load(handle)
    elif isinstance(filename, Path):
        with filename.open('rb') as handle:
            obj = pickle.load(handle)

    return obj


def save_pickle(obj, fname):
    if isinstance(fname, str):
        with open(fname, 'wb') as handle:
            pickle.dump(obj, handle, pickle.HIGHEST_PROTOCOL)
    elif isinstance(fname, Path):
        with fname.open('wb') as handle:
            pickle.dump(obj, handle, pickle.HIGHEST_PROTOCOL)
