# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

"""
Run benchmarks for SMAC.

Requires installing smac from pip. The experiments here used version 2.7.0.
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

from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from smac.configspace import ConfigurationSpace
from smac.runhistory.runhistory import RunKey
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.tae.execute_func import ExecuteTAFuncArray
from ConfigSpace.hyperparameters import UniformFloatHyperparameter


def fmin_smac_nopynisher(func, x0, bounds, maxfun, rng):
    """
    Minimize a function using SMAC, but without pynisher, which doesn't work
    well with benchmark_minimize_callable.
    
    This function is based on SMAC's fmin_smac.
    """
    cs = ConfigurationSpace()
    tmplt = 'x{0:0' + str(len(str(len(bounds)))) + 'd}'
    for idx, (lower_bound, upper_bound) in enumerate(bounds):
        parameter = UniformFloatHyperparameter(
            name=tmplt.format(idx + 1),
            lower=lower_bound,
            upper=upper_bound,
            default_value=x0[idx],
        )
        cs.add_hyperparameter(parameter)

    scenario_dict = {
        "run_obj": "quality",
        "cs": cs,
        "deterministic": "true",
        "initial_incumbent": "DEFAULT",
        "runcount_limit": maxfun,
    }
    scenario = Scenario(scenario_dict)

    def call_ta(config):
        x = np.array([val for _, val in sorted(config.get_dictionary().items())],
                     dtype=np.float)
        return func(x)

    smac = SMAC4HPO(
        scenario=scenario,
        tae_runner=ExecuteTAFuncArray,
        tae_runner_kwargs={'ta': call_ta, 'use_pynisher': False},
        rng=rng,
        initial_design=RandomConfigurations,
    )

    smac.optimize()
    return


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
        method_name='smac',
        replication_index=rep,
    )

    fmin_smac_nopynisher(
        func=f,
        x0=[0.5] * D,
        bounds=[[0, 1]] * D,
        maxfun=200,
        rng=rep + 1,
    )

    rs_str = 'random_subspace_' if random_subspace else ''
    with open(f'results/hartmann6_{rs_str}{D}_smac_rep_{rep}.json', "w") as fout:
       json.dump(object_to_json(experiment), fout)


def run_branin_benchmarks(rep):

    experiment, f = benchmark_minimize_callable(
        problem=branin_100,
        num_trials=50,
        method_name='smac',
        replication_index=rep,
    )

    fmin_smac_nopynisher(
        func=f,
        x0=[2.5] * 50 + [7.5] * 50,
        bounds=[[-5, 10]] * 50 + [[0, 15]] * 50,
        maxfun=50,
        rng=rep + 1,
    )

    with open(f'results/branin_100_smac_rep_{rep}.json', "w") as fout:
       json.dump(object_to_json(experiment), fout)


if __name__ == '__main__':
    # Run all of the SMAC experiments.
    # These can be distributed.

    for i in range(50):
        # Hartmann6, D=100: Each rep takes ~1.5 hours
        run_hartmann6_benchmarks(D=100, rep=i)

        # Branin, D=100: Each rep takes ~20 mins
        run_branin_benchmarks(rep=i)

    # Hartmann6, D=1000: Each rep takes ~36 hours
    for i in range(10):
        run_hartmann6_benchmarks(D=1000, rep=i)

        # Hartmann6 random subspace, D=1000
        run_hartmann6_benchmarks(D=1000, rep=i, random_subspace=True)
