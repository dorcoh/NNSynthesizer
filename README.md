# About

This repository provides a novel framework to repair unsafe neural networks w.r.t. safety specification. It builds upon 
methods from the field of formal verification. More concretely, it uses SMT solvers to search for the repaired network. 

Practically, this framework takes as input a Neural Network and a safety specification plus a set of free weights, it 
then searches for a new weights assignment (if exists) that will repair the neural network (i.e., generate a new, safe NN).

The codebase is part of my **M.Sc. thesis**: `Automated Repair of Neural Networks`.

Additional resources regarding the thesis work can be found on this [page](http://intractable.netlify.com/research).


# Built With
- Python 3.7.10
- Z3 SMT solver 4.8.12 (+ Python wrapper)
- PyTorch 1.7.0

# Getting Started

## Installation
- Install Python and Z3
- Install required Python packages:

```bash
pip install requirements.txt
```

## Main flow - usage example

The `main.py` module is used for development, it basically demonstrates a single repair example, it takes a config file
for setting up and executing a single repair trial. It can also take command line arguments (config overrides).
 
For example, the following command will attempt to repair a network with the samples similarity heuristic:

```
python main.py --config config-example-samples.json
```

The flow works as follows:
1. Load a serialized dataset from the `datasets` directory (filename is controlled by `load_dataset` argument)
2. Load a serialized trained neural network from the `model` directory (filename is controlled by `load_nn` argument)
3. Build the desired property definition (controlled by `properties` argument)
4. Select the set of free weights (controlled by `weights_config` argument)
5. Select similarity heuristic (controlled by `heuristic` argument)
6. Generate an SMT formula according to: network topology and its original values, desired property and free weights
7. Save the SMT formula (for debug purposes)
8. Solve the formula (the Python wrapper executes Z3, timeout is controlled by `z3_timeout`)
9. In case repair has succeeded: save the fixed (serialized) network, evaluate its decision boundary and compute metrics.
10. Otherwise (repair failed): exit the flow.

##### Outputs
This flow stores its output files in `./results-main/{timestamp}/`, each run stores its results in a unique directory 
with the current unix timestamp. We store the full config, SMT formula, log, metrics sheet, the serialized fixed model, 
the new (and old) weights values, and a plot with the original and repaired decision boundaries plus a summary.


##### Additional configs (for using different similarity heuristics):
- `config-example-grid.json` - to run a demo repair with the grid similarity heuristic
- `config-example-voronoi.json` - to run a demo repair with the voronoi similarity heuristic


## Advanced usage examples

#### 1 - Running an experiment set (multiple repair trials)

To run an experiment set for repairing a network with our proposed method:

```
python repair_exp_runner.py --exp_config_path config-dev.json
```
The `config-dev.json` contains all the relevant information for running this experiment set. 
(e.g., network, properties, weight combinations, etc.). The flow will then attempt to repair the network according to 
each of the trials settings, in a similar manner to the main flow.

Note that the required config file is similar to the one described in the main flow section, however, general arguments 
should now be under the `global` scope, and the experiments related properties are set under `experiments` (which is a JSON list). 
In addition, hyper parameters can be set. For example, in `config-dev.json` under `experiments.hyper_params` we can observe multiple 
weight configs and threshold values. In practice, the flow will try to generate a repair trial from each combination of
weight config and threshold (e.g., if we have 4 weights configs and 3 threshold values, the flow will execute a total
 of 12 repair trials). **Please note - this config is required in all of the following advanced use cases.** 

Once the process is complete, the results should be available in `./results/{config_name}/{unix_timestamp}/{exp_i}` for
each experiment id `i`. Aggregated results shall be available in `./results/{config_name}/{unix_timestamp}/general_stats.csv`.

##### Pre-compute experiment set configs (caching)

To allow resuming the `repair_exp_runner` from failure, one can pre-compute the experiment config by following these steps:

1. Execute `python repair_exp_runner_cache.py --exp_config_path {the-desired-config-name}.json`
2. Make a directory for your experiment set under `./results/{the-desired-config-name}/{experiment-set-name}`
3. Copy `general_stats_cached.csv` from `./cache/{the-desired-config-name}/` to your experiment set folder
4. Rename the file (from 3) to `general_stats.csv`
5. Execute `python repair_exp_runner.py --exp_config_path {the-desired-config-name}.json --cache True --timestamp {experiment-set-name} --exp_id {exp-id-to-resume}`. 

For example, executing the following: `python repair_exp_runner.py --exp_config_path config-dev.json --cache True --timestamp test --exp_id 3` 
will attempt to resume exp_id `3` from the `config-dev` config name, and the `test` experiment set name.

#### 2 - Running an experiment set with the **naive baseline**

To run an experiment set for repairing the network with the **naive baseline** method (SGD based):
```bash
python naive_repair_sgd.py --exp_config_path config-xor-bad-8-samples-first-prop-2-weight-combs.json --epochs 1
```
Will attempt to repair the network using the naive method, according to the trials settings. Results shall be available
in `./results-sgd/config-xor-bad-8-samples-first-prop-2-weight-combs/{timestamp}/`.

#### 3 - Training a network

To train a neural network (requires a dataset):

```bash
python nn_trainer.py --load_dataset datasets/good-network-data.XorDataset.pkl --trainer_subset 0.1 --save_plot True --trained_nn_suffix my-cool-nn 
```

This will result in loading the dataset, sampling 10% of it (`trainer_subset`), and training the network according to it.
Lastly, the trained network will be stored as `./model-my-cool-nn.pkl`.

#### 4 - Serialize a trained network

To serialize a network instance:
```bash
python serialize_net.py --exp_config_path config-dev.json 
```

This will result with loading the network and property mentioned in `exp_config`, then saving a serialized instance 
of the network representation (weights) and the property, at `serializer_results/config-dev/{timestamp}/model.json`.

#### 5 - Inspect a (serialized) network and its desired property
   
To inspect a network and property (with plot):
```bash
python inspect_net_and_property.py --exp_config_path config-dev.json --plot_name XOR-A 
```

This will result with loading the network and its property, generate a corresponding SMT formula and solve it, and 
evaluating the network decision boundaries (by plot). Formula and plot shall be available at `./inspector_results/{timestamp}/`

### Other modules and scripts
- Main logic: `nnsynth` module
- Dataset generation: `dataset_generator.py`


### Reproduce thesis experiments
To reproduce our reported experiments you should follow example (1) above, by running each of these experiment sets
under `./exp_configs`:


##### Exp 1 - Compare similarity-preserving heuristics

```bash
config-xor-bad-1.json
config-xor-bad-2.json
config-xor-bad-3.json
config-xor-bad-4-one-weight-combs.json
config-xor-bad-5-one-weight-combs.json
```

##### Exp 2 - Repair with Samples similarity heuristic

###### XOR-B network
```bash
config-xor-bad-6.json
config-xor-bad-7.json
config-xor-bad-8-samples-first-prop-2-weight-combs.json
config-xor-bad-9-samples-second-prop-2-weight-combs.json
config-xor-bad-10-2props-1w.json
config-xor-bad-11-2props-2w.json
```

###### Blobs network
```bash
config-blobs-1.json
config-blobs-2.json
config-blobs-3.json
```

##### Exp 3 - Compare to naive baseline
Then, you should also follow example (2) above to experiment with the naive method, these are the relevant configs:

###### XOR-B network

```bash
config-xor-bad-8-samples-first-prop-2-weight-combs.json
config-xor-bad-9-samples-second-prop-2-weight-combs.json
config-xor-bad-10-2props-1w.json
```

###### Blobs network
```bash
config-blobs-1.json
config-blobs-2.json
config-blobs-3.json
```

# Experiments Results

The raw experiments results are available at `./processed-results/combined-1.csv` for the first part of experiments 
(comparing similarity heuristics) and `./processed-results/combined-2.csv` for the second part (repair `XOR-B` and 
`Blobs` networks with the samples similarity heuristic). Finally, the **naive baseline** results are available under 
`./results-sgd/`.

# Citation

TODO

# Issues and contact

Feel free to open issue threads, you can also connect with me on [LinkedIn](https://www.linkedin.com/in/dorcoh).