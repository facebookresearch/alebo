# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

"""
Run benchmarks for Add-GP-UCB.

Requires installing dragonfly-opt from pip. The experiments here used version
0.1.4.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from argparse import Namespace
import json

from benchmark_problems import (
    branin_100,
    hartmann6_100,
    hartmann6_1000,
)

from ax.benchmark.benchmark import benchmark_minimize_callable
from ax.storage.json_store.encoder import object_to_json
from dragonfly import minimise_function  # dragonfly-opt==0.1.4

def run_hartmann6_benchmarks(D, rep):
    if D == 100:
        problem = hartmann6_100
    elif D == 1000:
        problem = hartmann6_1000

    experiment, f = benchmark_minimize_callable(
        problem=problem,
        num_trials=200,
        method_name='add_gp_ucb',
        replication_index=rep,
    )

    options = Namespace(acq="add_ucb")
    res = minimise_function(f, domain=[[0, 1]] * D, max_capital=199, options=options)

    with open(f'results/hartmann6_{D}_addgpucb_rep_{rep}.json', "w") as fout:
       json.dump(object_to_json(experiment), fout)


def run_branin_benchmarks(rep):

    experiment, f = benchmark_minimize_callable(
        problem=branin_100,
        num_trials=50,
        method_name='add_gp_ucb',
        replication_index=rep,
    )

    options = Namespace(acq="add_ucb")
    res = minimise_function(
        f,
        domain=[[-5, 10]] * 50 + [[0, 15]] * 50,
        max_capital=49,
        options=options,
    )

    with open(f'results/branin_100_addgpucb_rep_{rep}.json', "w") as fout:
       json.dump(object_to_json(experiment), fout)


if __name__ == '__main__':
    # Run all of the Add-GP-UCB benchmarks using Dragonfly.
    # These can be distributed.

    for i in range(50):
        run_hartmann6_benchmarks(D=100, rep=i)

        ## Hartmann6, D=1000: Too slow, not run
        #run_hartmann6_benchmarks(D=1000, rep=i)

        # Branin, D=100: Each rep takes ~3 hours
        run_branin_benchmarks(rep=i)
