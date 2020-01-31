# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

"""
Run benchmarks for TuRBO.

Requires installing turbo from https://github.com/uber-research/TuRBO.

The experiments here used version 0.0.1 (commit 8461f9c).
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import json
import numpy as np

from benchmark_problems import (
    branin_100,
    hartmann6_100,
    hartmann6_1000,
    hartmann6_random_subspace_1000,
)

from ax.benchmark.benchmark import benchmark_minimize_callable
from ax.storage.json_store.encoder import object_to_json
import turbo


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
        method_name='turbo',
        replication_index=rep,
    )

    turbo1 = turbo.Turbo1(
        f=f,
        lb=np.zeros(D),
        ub=np.ones(D),
        n_init=10,
        max_evals=200,
    )

    turbo1.optimize()

    rs_str = 'random_subspace_' if random_subspace else ''
    with open(f'results/hartmann6_{rs_str}{D}_turbo_rep_{rep}.json', "w") as fout:
       json.dump(object_to_json(experiment), fout)


def run_branin_benchmarks(rep):

    experiment, f = benchmark_minimize_callable(
        problem=branin_100,
        num_trials=50,
        method_name='turbo',
        replication_index=rep,
    )

    turbo1 = turbo.Turbo1(
        f=f,
        lb=np.hstack((-5 * np.ones(50), np.zeros(50))),
        ub=np.hstack((10 * np.ones(50), 15 * np.ones(50))),
        n_init=10,
        max_evals=50,
    )

    turbo1.optimize()

    with open(f'results/branin_100_turbo_rep_{rep}.json', "w") as fout:
       json.dump(object_to_json(experiment), fout)


if __name__ == '__main__':
    # Run all of the TuRBO experiments.
    # These can be distributed.

    for i in range(50):
        # Hartmann6, D=100: Each rep takes ~15 mins 
        run_hartmann6_benchmarks(D=100, rep=i)

        # Hartmann6, D=1000: Each rep takes ~30 mins
        run_hartmann6_benchmarks(D=1000, rep=i)

        # Branin, D=100: Each rep takes ~5 mins
        run_branin_benchmarks(rep=i)

        # Hartmann6 random subspace, D=1000
        run_hartmann6_benchmarks(D=1000, rep=i, random_subspace=True)
