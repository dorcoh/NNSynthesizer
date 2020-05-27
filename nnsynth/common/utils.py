import os
import pickle
from pathlib import Path
from typing import Dict


def serialize_exp(input_size, num_classes, num_layers, coefs, intercepts, eval_set, experiment, filename_suffix='ser.exp'):
    serialization_dict = {
        'experiment': experiment,
        'input_size': input_size,
        'num_classes': num_classes,
        'num_layers': num_layers,
        'coefs': coefs,
        'intercepts': intercepts,
        'eval_set': eval_set
    }
    exp_path = Path('exp')
    if not exp_path.exists():
        exp_path.mkdir()

    filename_suffix = experiment + '_' + filename_suffix
    path = exp_path / filename_suffix
    save_pickle(serialization_dict, path)


def deserialize_exp(experiment='', filename_suffix='ser.exp'):
    exp_path = Path('exp')
    filename_suffix = experiment + '_' + filename_suffix
    path = exp_path / filename_suffix
    return load_pickle(path)


def serialize_main_loop_instance(weight_tuple, threshold, eval_set_size, filename):
    serialization_dict = {
        'weight_comb': weight_tuple,
        'threshold': threshold,
        'eval_set_size': eval_set_size
    }

    sub_exp_path = Path('sub-exp')
    if not sub_exp_path.exists():
        sub_exp_path.mkdir()
    path = sub_exp_path / filename
    print("Serializing main loop instance : {}".format(filename))
    save_pickle(serialization_dict, path)


def parse_weight_comb(weight_comb: tuple):
    """Parse weight combination tuple to string"""
    s = ""
    for i, elem in enumerate(weight_comb):
        s += str(elem)
        if i != len(weight_comb)-1:
            s += '-'
    return s


def save_exp_details(model_config: Dict, result, distance, model_mapping, filename=None):
    results_path = Path('repair-results')
    if not results_path.exists():
        results_path.mkdir()

    model_config['result'] = str(result)
    model_config['distance'] = distance
    model_config['mapping'] = model_mapping
    if filename is None:
        results_file_name = 'result_dict_' + parse_weight_comb(model_config['weight_comb']) + '_t-' + str(model_config['threshold'])
    else:
        results_file_name = filename

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
