# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

"""
Run NASBench benchmarks
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import json
import numpy as np

# This loads the nasbench dataset, which takes ~30s
from nasbench_evaluation import (
    get_nasbench_ax_client,
    evaluate_parameters,
    NASBenchRunner,
)

from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.modelbridge.strategies.rembo import HeSBOStrategy, REMBOStrategy
from ax.storage.json_store.encoder import object_to_json
from ax.core.data import Data

import turbo
import cma


def run_nasbench_benchmarks_ax(rep):
    """
    Runs the Ax methods on the nasbench benchmark
    (Sobol, ALEBO, HeSBO, REMBO)
    """
    gs_list = [
        GenerationStrategy(name="Sobol", steps=[GenerationStep(model=Models.SOBOL, num_trials=-1)]),
        ALEBOStrategy(D=36, d=12, init_size=10),
        HeSBOStrategy(D=36, d=12, init_per_proj=10),
        REMBOStrategy(D=36, d=12, init_per_proj=4),
    ]
    for gs in gs_list:
        try:
            axc = get_nasbench_ax_client(gs)
            for i in range(50):
                param_dict_i, trial_index = axc.get_next_trial()
                raw_data = evaluate_parameters(param_dict_i)
                axc.complete_trial(trial_index=trial_index, raw_data=raw_data)
            with open(f'results/nasbench_{gs.name}_rep_{rep}.json', 'w') as fout:
                json.dump(object_to_json(axc.experiment), fout)
        except Exception:
            pass
    return


def run_nasbench_benchmarks_turbo(rep):
    r = NASBenchRunner(max_eval=50)
    turbo1 = turbo.Turbo1(
        f=r.f,
        lb=np.zeros(36),
        ub=np.ones(36),
        n_init=10,
        max_evals=50,
        batch_size=1,
    )
    turbo1.optimize()
    with open(f'results/nasbench_turbo_rep_{rep}.json', "w") as fout:
       json.dump((r.fs, r.feas), fout)


def run_nasbench_benchmarks_cmaes(rep):
    r = NASBenchRunner(max_eval=50)
    try:
        cma.fmin(
            objective_function=r.f,
            x0=[0.5] * 36,
            sigma0=0.25,
            options={
                'bounds': [[0.0] * 36, [1.0] * 36],
                'maxfevals': 50,
            },
        )
    except ValueError:
        pass  # CMA-ES doesn't always terminate at exactly maxfevals
    with open(f'results/nasbench_cmaes_rep_{rep}.json', "w") as fout:
       json.dump((r.fs, r.feas), fout)


if __name__ == '__main__':
    for rep in range(100):
        run_nasbench_benchmarks_cmaes(rep)
        run_nasbench_benchmarks_turbo(rep)
        run_nasbench_benchmarks_notbatch(rep)
