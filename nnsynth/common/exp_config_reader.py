"""Provide the functionality for reading configurations which are intended for `repair_exp_runner.py`."""
import itertools
import json
import logging
from copy import copy
from pathlib import Path
from typing import Dict, List

import pandas as pd


class ExpConfigReader:
    def __init__(self, path: Path):
        with path.open('r') as handle:
            self.config: Dict = json.load(handle)

    def parse_experiments(self, _args=None):
        for exp in self.config.get('experiments'):
            exp_settings: Dict = exp['settings']
            comb_dicts = self.explode_hyper_params(exp['hyper_params'])
            for comb in comb_dicts:
                args = copy(exp_settings)
                args.update(**comb)
                logging.debug(f"curr args for exp: {args}")
                yield args

    def explode_hyper_params(self, hyperparams_dict):
        """Turns hyper_params dict into a list of combinations"""
        comb_tuples = list(itertools.product(*hyperparams_dict.values()))
        keys = list(hyperparams_dict.keys())
        comb_dicts: List[Dict] = []
        for tup in comb_tuples:
            curr = {}
            for i, key in enumerate(keys):
                curr[key] = tup[i]
            comb_dicts.append(curr)

        return comb_dicts

    def _get_experiments_instances(self, args=None):
        return self.parse_experiments(args)

    def _get_cached_experiment_instances(self, args, path):
        # based on `general_stats.csv` file
        df = pd.read_csv(path)
        # exp_id start at 1, while idx start at 0
        for idx, row in df.iloc[args.exp_id-1:].iterrows():
            cfg_path = Path(row['config_path'])
            with cfg_path.open('r') as handle:
                exp_cfg: Dict = json.load(handle)
                yield idx, exp_cfg

    def get_experiments_instances(self, args=None, path=None):
        logging.info("get_experiments_instances")
        if not args.cache:
            logging.info("get_experiments_instances::regular mode")
            return enumerate(self._get_experiments_instances())
        else:
            logging.info(f"get_experiments_instances::cache mode. params: ts: {args.timestamp}, exp_id: {args.exp_id}")
            return self._get_cached_experiment_instances(args, path)

    def update_args_with_global_config(self, args):
        vars(args).update(**self.config['global'])

    def get_config_dict(self):
        return self.config