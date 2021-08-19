import json
import os
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd

EXP_FILENAME_SUFFIX = '.exp.pkl'

def serialize_exp(input_size, num_classes, num_layers, coefs, intercepts, eval_set, experiment, hidden_size, property,
                  filename_suffix=EXP_FILENAME_SUFFIX):
    serialization_dict = {
        'experiment': experiment,
        'input_size': input_size,
        'num_classes': num_classes,
        'num_layers': num_layers,
        'coefs': coefs,
        'intercepts': intercepts,
        'eval_set': eval_set,
        'hidden_size': hidden_size,
        'property': property
    }
    exp_path = Path('exp')
    if not exp_path.exists():
        exp_path.mkdir()

    filename = experiment + filename_suffix
    path = exp_path / filename
    save_pickle(serialization_dict, path)


def deserialize_exp(experiment='', filename_suffix=EXP_FILENAME_SUFFIX):
    exp_path = Path('exp')
    filename = experiment + filename_suffix
    path = exp_path / filename
    return load_pickle(path)


def deserialize_subexp(experiment, sub_experiment):
    sub_exp_path = Path('sub-exp')
    path = sub_exp_path / experiment / sub_experiment
    return load_pickle(path)

def serialize_main_loop_instance(weight_tuple, threshold, eval_set_size, filename, experiment):
    serialization_dict = {
        'weight_comb': weight_tuple,
        'threshold': threshold,
        'eval_set_size': eval_set_size
    }

    sub_exp_path = Path('sub-exp/{}'.format(experiment))
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


def save_exp_details(model_config: Dict, result, distance, model_mapping, experiment, filename=None):
    results_path = Path('repair-results/{}'.format(experiment))
    results_path.mkdir(parents=True, exist_ok=True)

    model_config['result'] = str(result)
    model_config['distance'] = distance
    model_config['mapping'] = model_mapping
    if filename is None:
        # TODO: remove this case (supported on main_loop.py only?)
        results_file_name = 'result_dict_' + parse_weight_comb(model_config['weight_comb']) + '_t-' + str(model_config['threshold'])
    else:
        results_file_name = filename

    path_to_store = results_path / results_file_name
    print("saving exp results pickle in: {}".format(str(path_to_store.absolute().as_uri())))
    save_pickle(model_config, results_path / results_file_name)
    # TODO: append to some tabular structure, in addition to single files (so we could quickly aggregate the stats)


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

def save_exp_config(args: Dict, path: Path):
    with path.open('w') as handle:
        json.dump(args, handle, indent=4)


def append_stats(path, exp_id, exp_key, metrics, time_took, extra=None):
    """Append general stats (aggregating all experiments)"""
    df = pd.DataFrame([{
        'exp_id': exp_id,
        'exp_key': exp_key,
        'extra_params': extra,
        'avg_acc_before': metrics['original_avg'] if 'original_avg' in metrics else None,
        'avg_acc_after': metrics['repaired_avg'] if 'repaired_avg' in metrics else None,
        'time': time_took}])

    with open(path, 'a') as f:
        df.to_csv(f, header=f.tell() == 0, index=False)