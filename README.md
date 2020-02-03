# ALEBO
This is the code associated with the paper "[Re-Examining Linear Embeddings for High-Dimensional Bayesian 
Optimization](https://arxiv.org/abs/2001.11659)"

If you find this code useful please cite it as

    @article{Letham2019Re,
        author    = {Letham, Benjamin and Calandra, Roberto and Rai, Akshara and Bakshy, Eytan},        
        title     = {Re-Examining Linear Embeddings for High-dimensional Bayesian Optimization},
        journal   = {arXiv preprint arXiv: 2001.11659},
        year      = {2020},
    }

## Installation
To install the code clone the repo and install the dependencies as

    git clone https://github.com/facebookresearch/alebo.git 
	cd alebo
    pip install -r requirements.txt
    
Some of the baselines require additional packages that can not be pip-installed. 
Detailed instructions can be found inside each file of the `benchmarks/` folder.

## Using ALEBO for optimizing a function
See `quickstart.ipynb` for a simple example of how to use ALEBO to optimize a function. ALEBO is built using the [Ax platform](https://ax.dev/); see instructions there on how to install via pip. You will need version 0.1.9.

## Reproducing the experiments
This repository contains the code required to run the benchmark experiments and generate the figures in the paper. The only exception are the DAISY figures, since the simulator is not yet open source.

### Generating figures:
The `figs/` directory contains a file to generate each of the figures in the paper, as indicated by the file name. Some figures show the results of simulations; in these cases the file contains code to both run the simulation and create the figure. For example, executing `figs/fig_4.py` will run the P_opt simulation described in the paper, will store the simulation results in `figs/data/`, and will then generate the figure based on those results. The pdf for Fig. 4 will be saved in `figs/pdfs/`.

### Running benchmark experiments
The `benchmarks/` directory contains code for running the benchmark BO experiments described in the paper. The benchmark problems are defined in `benchmark_problems.py`. Each method has its own script for evaluating that method on the appropriate set of benchmark problems: `run_{method}_benchmarks.py`, where `{method}` is:

* `ax`, for our implementations of ALEBO, HeSBO, and REMBO
* `addgpucb` for Add-GP-UCB via Dragonfly
* `cmaes` for CMA-ES
* `ebo` for Ensemble Bayesian Optimization
* `linebo` for LineBO
* `smac` for SMAC
* `turbo` for TuRBO

See the paper for references for each of these methods. Each file explains what needs to be done in order to run the experiments for that method. For instance, `run_cmaes_benchmarks.py` requires installing `cma` from pip; `run_ebo_benchmarks.py` requires cloning a repository. See each file for its instructions.

The file `run_rrembo_benchmarks.R` provides a similar script in R for running the benchmark experiments for k-\Psi REMBO variants. These use the R package `RRembo`, and results are stored in json.

All benchmark results are stored in `benchmark/results/` (the json files produced by each run of each method are not shipped in this repo). Once all of the `run_*_benchmarks.*` files have been run, `compile_benchmark_results.py` is used to compile the results from all of the different methods into a single file for each experiment. These files are `benchmarks/results/*_aggregated_results.json` and are included in this repository as the benchmark results used in the paper.

Executing `figs/fig_5.py` loads these aggregated results and generates the benchmark results figure in the paper.

### The ALEBO model and generation code
The actual implementation of the ALEBO method is at: https://github.com/facebook/Ax/blob/master/ax/models/torch/alebo.py

## License
This code is licensed under CC-by-NC, as found in the LICENSE file.
