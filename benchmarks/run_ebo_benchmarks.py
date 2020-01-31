# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

"""
Run benchmarks for Ensemble BO.

A few steps are required to use EBO:

(1) git clone https://github.com/zi-w/Ensemble-Bayesian-Optimization
 in this directory. These experiments used commit
 4e6f9ed04833cc2e21b5906b1181bc067298f914.

(2) fix a python3 issue by editing ebo_core/helper.py to insert
     shape = int(shape)
 in line 7.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(1, os.path.join(os.getcwd(), 'Ensemble-Bayesian-Optimization'))
from ebo_core.ebo import ebo

import json
import numpy as np

from benchmark_problems import (
    branin_100,
    hartmann6_100,
    hartmann6_1000,
)

from ax.benchmark.benchmark import benchmark_minimize_callable
from ax.storage.json_store.encoder import object_to_json


# These options are taken as package defaults from test_ebo.py
core_options = {
    #'x_range':x_range, # input domain
    #'dx':x_range.shape[1], # input dimension
    #'max_value':f.f_max + sigma*5, # target value
    #'T':10, # number of iterations
    'B':10, # number of candidates to be evaluated
    'dim_limit':3, # max dimension of the input for each additive function component
    'isplot':0, # 1 if plotting the result; otherwise 0. 
    'z':None, 'k':None, # group assignment and number of cuts in the Gibbs sampling subroutine
    'alpha':1., # hyperparameter of the Gibbs sampling subroutine
    'beta':np.array([5.,2.]), 
    'opt_n':1000, # points randomly sampled to start continuous optimization of acfun
    'pid':'test3', # process ID for Azure
    'datadir':'tmp_data/', # temporary data directory for Azure
    'gibbs_iter':10, # number of iterations for the Gibbs sampling subroutine
    'useAzure':False, # set to True if use Azure for batch evaluation
    'func_cheap':True, # if func cheap, we do not use Azure to test functions
    'n_add':None, # this should always be None. it makes dim_limit complicated if not None.
    'nlayers': 100, # number of the layers of tiles
    'gp_type':'l1', # other choices are l1, sk, sf, dk, df
    #'gp_sigma':0.1, # noise standard deviation
    'n_bo':10, # min number of points selected for each partition
    'n_bo_top_percent': 0.5, # percentage of top in bo selections
    'n_top':10, # how many points to look ahead when doing choose Xnew
    'min_leaf_size':10, # min number of samples in each leaf
    'max_n_leaves':10, # max number of leaves
    'thresAzure':1, # if batch size > thresAzure, we use Azure
    'save_file_name': 'tmp/tmp.pk',
}


def run_hartmann6_benchmarks(D, rep):
    if D == 100:
        problem = hartmann6_100
    elif D == 1000:
        problem = hartmann6_1000

    experiment, f = benchmark_minimize_callable(
        problem=problem,
        num_trials=200,
        method_name='ebo',
        replication_index=rep,
    )

    options = {
        'x_range': np.vstack((np.zeros(D), np.ones(D))),
        'dx': D,
        'max_value': 3.32237,  # Let it cheat and know the true max value
        'T': 200,
        'gp_sigma': 1e-7,
    }
    options.update(core_options)

    f_max = lambda x: -f(x)  # since EBO maximizes

    e = ebo(f_max, options)
    try:
        e.run()
    except ValueError:
        pass  # EBO can ask for more than T function evaluations

    with open(f'results/hartmann6_{D}_ebo_rep_{rep}.json', "w") as fout:
       json.dump(object_to_json(experiment), fout)


def run_branin_benchmarks(rep):

    experiment, f = benchmark_minimize_callable(
        problem=branin_100,
        num_trials=50,
        method_name='ebo',
        replication_index=rep,
    )

    options = {
        'x_range': np.vstack((
            np.hstack((-5 * np.ones(50), np.zeros(50))),
            np.hstack((10 * np.ones(50), 15 * np.ones(50))),
        )),
        'dx': 100,
        'max_value': -0.397887,  # Let it cheat and know the true max value
        'T': 50,
        'gp_sigma': 1e-7,
    }
    options.update(core_options)

    f_max = lambda x: -f(x)  # since EBO maximizes

    e = ebo(f_max, options)
    try:
        e.run()
    except ValueError:
        pass  # EBO can ask for more than T function evaluations

    with open(f'results/branin_100_ebo_rep_{rep}.json', "w") as fout:
       json.dump(object_to_json(experiment), fout)


if __name__ == '__main__':
    # Run all of the EBO benchmarks.
    # These can be distributed.

    for i in range(50):
        # Hartmann6, D=100: Each rep takes ~2 hours
        run_hartmann6_benchmarks(D=100, rep=i)

        ## Hartmann6, D=1000: Too slow, not run
        #run_hartmann6_benchmarks(D=1000, rep=i)

        # Branin, D=100: Each rep takes ~20 mins
        run_branin_benchmarks(rep=i)
