# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

"""
Run benchmarks for: ALEBO, REMBO, HeSBO, and Sobol.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import json

from benchmark_problems import (
    branin_100,
    branin_by_D,
    gramacy_100,
    hartmann6_100,
    hartmann6_1000,
    hartmann6_random_subspace_1000,
)

from ax.benchmark.benchmark import full_benchmark_run
from ax.benchmark.benchmark_result import aggregate_problem_results
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.modelbridge.strategies.rembo import HeSBOStrategy, REMBOStrategy
from ax.storage.json_store.encoder import object_to_json


def run_hartmann6_benchmarks(D, rep, random_subspace=False):
    if D == 100:
        problem = hartmann6_100
    elif D == 1000 and not random_subspace:
        problem = hartmann6_1000
    elif D == 1000 and random_subspace:
        problem = hartmann6_random_subspace_1000

    strategy0 = GenerationStrategy(
        name="Sobol",
        steps=[
            GenerationStep(
                model=Models.SOBOL, num_arms=-1, model_kwargs={'seed': rep + 1}
            )
        ],
    )
    strategy1 = ALEBOStrategy(D=D, d=12, init_size=10)
    strategy2 = REMBOStrategy(D=D, d=6, init_per_proj=2)
    strategy3 = HeSBOStrategy(D=D, d=6, init_per_proj=10, name=f"HeSBO, d=d")
    strategy4 = HeSBOStrategy(D=D, d=12, init_per_proj=10, name=f"HeSBO, d=2d")

    all_benchmarks = full_benchmark_run(
        num_replications=1,  # Running them 1 at a time for distributed
        num_trials=200,
        batch_size=1,
        methods=[strategy0, strategy1, strategy2, strategy3, strategy4],
        problems=[problem],
    )

    rs_str = 'random_subspace_' if random_subspace else ''
    with open(
        f'results/hartmann6_{rs_str}{D}_alebo_rembo_hesbo_sobol_rep_{rep}.json', "w"
    ) as fout:
       json.dump(object_to_json(all_benchmarks), fout)


def run_branin_and_gramacy_100_benchmarks(rep):
    strategy0 = GenerationStrategy(
        name="Sobol",
        steps=[
            GenerationStep(
                model=Models.SOBOL, num_arms=-1, model_kwargs={'seed': rep + 1}
            )
        ],
    )
    strategy1 = ALEBOStrategy(D=100, d=4, init_size=10)
    strategy2 = REMBOStrategy(D=100, d=2, init_per_proj=2)
    strategy3 = HeSBOStrategy(D=100, d=4, init_per_proj=10, name=f"HeSBO, d=2d")

    all_benchmarks = full_benchmark_run(
        num_replications=1,
        num_trials=50,
        batch_size=1,
        methods=[strategy0, strategy1, strategy2, strategy3],
        problems=[branin_100, gramacy_100],
    )

    with open(
        f'results/branin_gramacy_100_alebo_rembo_hesbo_sobol_rep_{rep}.json', "w"
    ) as fout:
       json.dump(object_to_json(all_benchmarks), fout)


def run_sensitivity_D_benchmarks(rep):
    results_dict = {}
    for D, problem in branin_by_D.items():
        strategy1 = ALEBOStrategy(D=D, d=4, init_size=10)

        all_benchmarks = full_benchmark_run(
            num_replications=1,
            num_trials=50,
            batch_size=1,
            methods=[strategy1],
            problems=[problem],
        )

        results_dict[D] = object_to_json(all_benchmarks)

    with open(f'results/sensitivity_D_rep_{rep}.json', "w") as fout:
       json.dump(results_dict, fout)


def run_sensitivity_d_e_benchmarks(rep):
    strategies = [
        ALEBOStrategy(D=100, d=d_e, init_size=10, name=f'ALEBO, d={d_e}')
        for d_e in [2, 3, 5, 6, 7, 8]
    ]

    all_benchmarks = full_benchmark_run(
        num_replications=1,
        num_trials=50,
        batch_size=1,
        methods=strategies,
        problems=[branin_100],
    )

    with open(f'results/sensitivity_d_e_rep_{rep}.json', "w") as fout:
       json.dump(object_to_json(all_benchmarks), fout)


if __name__ == '__main__':
    # Run all of the benchmark replicates.
    # They are set up here to run as individual replicates becaus they can be
    # distributed.

    for i in range(50):
        # Hartmann6, D=100: Each rep takes ~2 hrs
        run_hartmann6_benchmarks(D=100, rep=i)

        # Hartmann6, D=1000: Each rep takes ~2.5 hrs
        run_hartmann6_benchmarks(D=1000, rep=i)

        # Hartmann6, D=1000: Each rep takes ~2.5 hrs
        run_hartmann6_benchmarks(D=1000, rep=i, random_subspace=True)

        # Branin and Gramacy, D=100: Each rep takes ~20 mins
        run_branin_and_gramacy_100_benchmarks(rep=i)

        # Sensitivity benchmarks: Each rep takes ~2 hrs
        run_sensitivity_D_benchmarks(rep=i)
        run_sensitivity_d_e_benchmarks(rep=i)
