# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

"""
Run benchmarks for CMAES.

Requires installing cma from pip. The experiments here used version 2.7.0.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import json

from benchmark_problems import (
    branin_100,
    hartmann6_100,
    hartmann6_1000,
    hartmann6_random_subspace_1000,
)

from ax.benchmark.benchmark import benchmark_minimize_callable
from ax.storage.json_store.encoder import object_to_json
import cma  # cma==2.7.0


def run_hartmann6_benchmarks(D, rep, random_subspace=False):
    if D == 100:
        problem = hartmann6_100
    elif D == 1000 and not random_subspace:
        problem = hartmann6_1000
    elif D == 1000 and random_subspace:
        problem = hartmann6_random_subspace_1000

    experiment, f = benchmark_minimize_callable(
        problem=problem,
        num_trials=200,
        method_name='cmaes',
        replication_index=rep,
    )

    try:
        cma.fmin(
            objective_function=f,
            x0=[0.5] * D,
            sigma0=0.25,
            options={'bounds': [[0] * D, [1] * D], 'maxfevals': 200},
        )
    except ValueError:
        pass  # CMA-ES doesn't always terminate at exactly maxfevals

    rs_str = 'random_subspace_' if random_subspace else ''
    with open(f'results/hartmann6_{rs_str}{D}_cmaes_rep_{rep}.json', "w") as fout:
       json.dump(object_to_json(experiment), fout)


def run_branin_benchmarks(rep):

    experiment, f = benchmark_minimize_callable(
        problem=branin_100,
        num_trials=50,
        method_name='cmaes',
        replication_index=rep,
    )

    try:
        cma.fmin(
            objective_function=f,
            x0=[2.5] * 50 + [7.5] * 50,
            sigma0=3.75,
            options={
                'bounds': [[-5] * 50 + [0] * 50, [10] * 50 + [15] * 50],
                'maxfevals': 50,
            },
        )
    except ValueError:
        pass  # CMA-ES doesn't always terminate at exactly maxfevals

    with open(f'results/branin_100_cmaes_rep_{rep}.json', "w") as fout:
       json.dump(object_to_json(experiment), fout)


if __name__ == '__main__':
    # Run all of the CMAES experiments.
    # These can be distributed.

    for i in range(50):
        # Hartmann6, D=100: Each rep takes ~5 s
        run_hartmann6_benchmarks(D=100, rep=i)

        # Hartmann6, D=1000: Each rep takes ~10 s
        run_hartmann6_benchmarks(D=1000, rep=i)

        # Branin, D=100: Each rep takes ~1 s
        run_branin_benchmarks(rep=i)

        # Hartmann6 random subspace, D=1000
        run_hartmann6_benchmarks(D=1000, rep=i, random_subspace=True)
