# About

This repository provides a novel framework to repair unsafe neural networks w.r.t. safety specification. It builds upon 
methods from the field of formal verification. More concretely, it uses SMT solvers to search for the repaired network. 

The codebase is part of my **M.Sc. thesis**: `Automated Repair of Neural Networks`.

More details regarding the thesis work (such as slides from the seminar) can be found on this [page](http://intractable.netlify.com/msc-thesis).
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

## Usage Examples

1. To run an experiment set for repairing a network with our proposed method:

    ```
    python repair_exp_runner.py --exp_config_path config-dev.json
    ```
    The `config-dev.json` contains all the relevant information for running this experiment set. 
    (e.g., network, properties, weight combinations, etc.)
    
    The script will attempt to repair the network according to the trials settings.
    
2. To run an experiment set for repairing the network with the **naive baseline** method (SGD based):
    ```bash
    python naive_repair_sgd.py --exp_config_path config-xor-bad-8-samples-first-prop-2-weight-combs.json --epochs 1
    ```
   Will attempt to repair the network using the naive method, according to the trials settings. 

3. To train a neural network (requires a dataset):
    ```bash
   python nn_trainer.py --load_dataset good-network-data.XorDataset.pkl --trainer_subset 0.1 --save_plot False 
   ```

4. To serialize a network instance:
    ```bash
   python serialize_net.py --exp_config_path config-dev.json 
   ```
   
5. To inspect a network and property (with plot):
    ```bash
   python inspect_net_and_property.py --exp_config_path config-dev.json --plot_name XOR-A 
   ```

### Other modules and scripts
- Main logic: `nnsynth` module
- Development module: `main.py` (represents a single repair trial)
- Generate actual config files (each equals to one repair trial) for experiment set: `repair_exp_runner_cache.py`
- Dataset generation: `dataset_generator.py`


### Reproduce Experiments
To reproduce our reported experiments you should follow example (1) above, by running each of these experiment sets:


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

# Citation

TODO

# Contact

You can connect with me on [LinkedIn](https://www.linkedin.com/in/dorcoh)