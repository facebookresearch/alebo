# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

"""
Compile all of the benchmark results from the different methods (potentially
run in a distributed fashion) into a single BenchmarkResult object.

All of the benchmark runs should be completed before running this.
"""

import gc
import json
import numpy as np

from benchmark_problems import (
    branin_100,
    branin_by_D,
    gramacy_100,
    hartmann6_100,
    hartmann6_1000,
    hartmann6_random_subspace_1000,
)
from ax.benchmark.benchmark_result import aggregate_problem_results
from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.decoder import object_from_json


def merge_benchmark_results(res1, res2):
    """
    Merges two benchmark results dictionaries in-place (res2 into res1)
    """
    for problem_name in res2:
        for method_name in res2[problem_name]:
            for exp in res2[problem_name][method_name]:
                res1 = add_exp(
                    res=res1,
                    exp=exp,
                    problem_name=problem_name,
                    method_name=method_name,
                )
    return res1


def add_exp(res, exp, problem_name, method_name):
    """
    Add benchmark experiment exp to results dict res, under the specified
    problem_name and method_name.
    """
    if problem_name not in res:
        res[problem_name] = {}
    if method_name not in res[problem_name]:
        res[problem_name][method_name] = []
    res[problem_name][method_name].append(exp)
    return res


def compile_hartmann6(D, random_subspace=False):
    if D == 100:
        problem = hartmann6_100
        other_methods = ['addgpucb', 'cmaes', 'ebo', 'smac', 'turbo']
        rls = ['rrembos_standard_kPsi', 'rrembos_reverse_kPsi', 'coordinatelinebo', 'descentlinebo', 'randomlinebo']
        rs_str = ''
    elif D == 1000 and not random_subspace:
        problem = hartmann6_1000
        other_methods = ['cmaes', 'smac', 'turbo']
        rls = ['rrembos_standard_kPsi']
        rs_str = ''
    elif D == 1000 and random_subspace:
        problem = hartmann6_random_subspace_1000
        other_methods = ['cmaes', 'smac', 'turbo']
        rls = []
        rs_str = 'random_subspace_'

    all_results = {}

    for rep in range(50):
        with open(f'results/hartmann6_{rs_str}{D}_alebo_rembo_hesbo_sobol_rep_{rep}.json', 'r') as fin:
            res_i = object_from_json(json.load(fin))

        all_results = merge_benchmark_results(all_results, res_i)

        for method_name in other_methods:
            if D==1000 and method_name == 'smac' and rep > 9:
                # SMAC D=1000 only run for 10 reps
                continue

            with open(f'results/hartmann6_{rs_str}{D}_{method_name}_rep_{rep}.json', 'r') as fin:
                exp_i = object_from_json(json.load(fin))

            all_results = add_exp(res=all_results, exp=exp_i, problem_name=problem.name, method_name=method_name)

    res = aggregate_problem_results(runs=all_results[problem.name], problem=problem)

    # Add in RRembo and LineBOresults
    for method in rls:
        with open(f'results/hartmann6_{D}_{method}.json', 'r') as fin:
            A = json.load(fin)
        res.objective_at_true_best[method] = np.minimum.accumulate(np.array(A), axis=1)

    # Save
    with open(f'results/hartmann6_{rs_str}{D}_aggregated_results.json', "w") as fout:
        json.dump(object_to_json({problem.name: res}), fout)


def compile_branin_gramacy_100():
    all_results = {}

    for rep in range(50):
        with open(f'results/branin_gramacy_100_alebo_rembo_hesbo_sobol_rep_{rep}.json', 'r') as fin:
            res_i = object_from_json(json.load(fin))

        all_results = merge_benchmark_results(all_results, res_i)

        for method_name in ['addgpucb', 'cmaes', 'ebo', 'smac', 'turbo']:
            with open(f'results/branin_100_{method_name}_rep_{rep}.json', 'r') as fin:
                exp_i = object_from_json(json.load(fin))

            all_results = add_exp(
                res=all_results,
                exp=exp_i,
                problem_name='Branin, D=100',
                method_name=method_name,
            )

    res = {
        p.name: aggregate_problem_results(runs=all_results[p.name], problem=p)
        for p in [branin_100, gramacy_100]
    }

    # Add in RRembo results
    for proj in ['standard', 'reverse']:
        method = f'rrembos_{proj}_kPsi'
        with open(f'results/branin_100_{method}.json', 'r') as fin:
            A = json.load(fin)
        res['Branin, D=100'].objective_at_true_best[method] = np.minimum.accumulate(np.array(A), axis=1)

    # Save
    with open(f'results/branin_gramacy_100_aggregated_results.json', "w") as fout:
        json.dump(object_to_json(res), fout)


def compile_sensitivity_benchmarks():
    all_results = {}

    for rep in range(50):
        ## Sensitivity to D
        with open(f'results/sensitivity_D_rep_{rep}.json', 'r') as fin:
            results_dict = json.load(fin)

        for D, obj in results_dict.items():
            res_i = object_from_json(obj)
            all_results = merge_benchmark_results(all_results, res_i)

        ## Sensitivity to d_e
        with open(f'results/sensitivity_d_e_rep_{rep}.json', 'r') as fin:
            res_i = object_from_json(json.load(fin))

        all_results = merge_benchmark_results(all_results, res_i)

    all_problems = (
        [hartmann6_100, hartmann6_1000, branin_100, gramacy_100]
        + list(branin_by_D.values())
    )

    problems = [branin_100] + list(branin_by_D.values())
    res = {
        p.name+'_sensitivity': aggregate_problem_results(runs=all_results[p.name], problem=p)
        for p in problems
    }
    # Save
    with open(f'results/sensitivity_aggregated_results.json', "w") as fout:
        json.dump(object_to_json(res), fout)


def compile_ablation_benchmarks():
    all_results = {}

    for rep in range(100):
        with open(f'results/ablation_rep_{rep}.json', 'r') as fin:
            res_i = object_from_json(json.load(fin))

        all_results = merge_benchmark_results(all_results, res_i)

    problems = [branin_100]
    res = {
        p.name+'_ablation': aggregate_problem_results(runs=all_results[p.name], problem=p)
        for p in problems
    }
    # Save
    with open(f'results/ablation_aggregated_results.json', "w") as fout:
        json.dump(object_to_json(res), fout)


def compile_nasbench():
    all_res = {}
    # TuRBO and CMAES
    for method in ['turbo', 'cmaes']:
        all_res[method] = []
        for rep in range(100):
            with open(f'results/nasbench_{method}_rep_{rep}.json', 'r') as fin:
                fs, feas = json.load(fin)
            # Set infeasible points to nan
            fs = np.array(fs)
            fs[~np.array(feas)] = np.nan
            all_res[method].append(fs)

    # Ax methods
    for method in ['Sobol', 'ALEBO', 'HeSBO', 'REMBO']:
        all_res[method] = []
        for rep in range(100):
            with open(f'results/nasbench_{method}_rep_{rep}.json', 'r') as fin:
                exp = object_from_json(json.load(fin))
            # Pull out results and set infeasible points to nan
            df = exp.fetch_data().df.sort_values(by='arm_name')
            df_obj = df[df['metric_name'] == 'final_test_accuracy'].copy().reset_index(drop=True)
            df_con = df[df['metric_name'] == 'final_training_time'].copy().reset_index(drop=True)
            infeas = df_con['mean'] > 1800
            df_obj.loc[infeas, 'mean'] = np.nan
            all_res[method].append(df_obj['mean'].values)

    for method, arr in all_res.items():
        all_res[method] = np.fmax.accumulate(np.vstack(all_res[method]), axis=1)

    with open(f'results/nasbench_aggregated_results.json', "w") as fout:
        json.dump(object_to_json(all_res), fout)


if __name__ == '__main__':
    compile_nasbench()
    gc.collect()
    compile_hartmann6(D=100)
    gc.collect()
    compile_hartmann6(D=1000)
    gc.collect()
    compile_branin_gramacy_100()
    gc.collect()
    compile_sensitivity_benchmarks()
    gc.collect()
    compile_hartmann6(D=1000, random_subspace=True)
    gc.collect()
    compile_ablation_benchmarks()
