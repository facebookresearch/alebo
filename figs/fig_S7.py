# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import json
import numpy as np

from ax.storage.json_store.decoder import object_from_json

from plot_config import *


def make_fig_S7():
    # Load in the benchmark results
    res = {}
    for fname in [
        'hartmann6_1000',
        'branin_gramacy_100',
        'hartmann6_100',
        'hartmann6_random_subspace_1000',
    ]:
        with open(f'../benchmarks/results/{fname}_aggregated_results.json', 'r') as fin:
            res.update(object_from_json(json.load(fin)))

    # A map from method idx in plot_method_names to the name used in res
    method_idx_to_res_name = {
        0: 'ALEBO',
        1: 'REMBO',
        2: 'HeSBO, d=d',
        3: 'HeSBO, d=2d',
        4: 'rrembos_standard_kPsi',
        5: 'rrembos_reverse_kPsi',
        6: 'ebo',
        7: 'addgpucb',
        8: 'smac',
        9: 'cmaes',
        10: 'turbo',
        11: 'Sobol',
        12: 'coordinatelinebo',
        13: 'randomlinebo',
        14: 'descentlinebo',
    }

    # Make the figure
    fig = plt.figure(figsize=(5.5, 7.5))

    ####### Branin, D=100
    ax1 = fig.add_subplot(511)

    res_h = res['Branin, D=100']

    for idx, m in enumerate(plot_method_names):
        res_name = method_idx_to_res_name[idx]
        if res_name not in res_h.objective_at_true_best:
            continue  # Not run on this problem
        Y = np.log(res_h.objective_at_true_best[res_name] - 0.397887)
        f = Y.mean(axis=0)
        sem = Y.std(axis=0) / np.sqrt(Y.shape[0])
        x = np.arange(1, 51)
        color = plot_colors[m]
        ax1.plot(x, f, color=color, label=m)
        ax1.errorbar(x[4::5], f[4::5], yerr=2 * sem[4::5], color=color, alpha=0.5, ls='')

    ax1.set_xlim([0, 51])
    ax1.set_ylim([-6, 2])
    ax1.set_ylabel('Log regret', fontsize=9)
    ax1.grid(alpha=0.2, zorder=-10)
    ax1.set_title(r'Branin, $d$=2, $D$=100')

    ####### Hartmann6, D=1000
    ax2 = fig.add_subplot(512)

    res_h = res['Hartmann6, D=1000']

    for idx, m in enumerate(plot_method_names):
        res_name = method_idx_to_res_name[idx]
        if res_name not in res_h.objective_at_true_best:
            continue  # Not run on this problem
        Y = np.log(res_h.objective_at_true_best[res_name] - (-3.32237))
        f = Y.mean(axis=0)
        sem = Y.std(axis=0) / np.sqrt(Y.shape[0])
        x = np.arange(1, 201)
        color = plot_colors[m]
        ax2.plot(x, f, color=color, label=m)
        ax2.errorbar(x[9::10], f[9::10], yerr=2 * sem[9::10], color=color, alpha=0.5, ls='')

    ax2.set_xlim([0, 201])
    ax2.set_ylim([-2.5, 1.7])
    ax2.set_ylabel('Log regret', fontsize=9)
    ax2.grid(alpha=0.2, zorder=-10)
    ax2.set_title(r'Hartmann6, $d$=6, $D$=1000')

    ####### Gramacy, D=100
    ax3 = fig.add_subplot(513)

    res_h = res['Gramacy, D=100']

    for idx, m in enumerate(plot_method_names):
        res_name = method_idx_to_res_name[idx]
        if res_name not in res_h.objective_at_true_best:
            continue  # Not run on this problem
        Y = np.log(res_h.objective_at_true_best[res_name] - 0.5998)
        f = Y.mean(axis=0)
        sem = Y.std(axis=0) / np.sqrt(Y.shape[0])
        x = np.arange(1, 51)
        color = plot_colors[m]
        ax3.plot(x, f, color=color, label=m)
        ax3.errorbar(x[4::5], f[4::5], yerr=2 * sem[4::5], color=color, alpha=0.5, ls='')
    
    ax3.set_xlim([0, 51])
    ax3.set_ylabel('Log regret', fontsize=9)
    ax3.grid(alpha=0.2, zorder=-10)
    ax3.set_title(r'Gramacy, $d$=2, $D=100$')

    ####### Hartmann6, D=100
    ax4 = fig.add_subplot(514)

    res_h = res['Hartmann6, D=100']

    for idx, m in enumerate(plot_method_names):
        res_name = method_idx_to_res_name[idx]
        if res_name not in res_h.objective_at_true_best:
            continue  # Not run on this problem
        Y = np.log(res_h.objective_at_true_best[res_name] - (-3.32237))
        f = Y.mean(axis=0)
        sem = Y.std(axis=0) / np.sqrt(Y.shape[0])
        x = np.arange(1, 201)
        color = plot_colors[m]
        ax4.plot(x, f, color=color, label=m)
        ax4.errorbar(x[9::10], f[9::10], yerr=2 * sem[9::10], color=color, alpha=0.5, ls='')

    ax4.set_xlim([0, 201])
    ax4.set_ylim([-4, 1.7])
    ax4.set_ylabel('Log regret', fontsize=9)
    ax4.grid(alpha=0.2, zorder=-10)
    ax4.set_title(r'Hartmann6, $d$=6, $D$=100')

    # Add the legend
    ax4.legend(bbox_to_anchor=(1.44, 5.405), fontsize=8)

    ####### Hartmann6 random subspace, D=1000
    ax5 = fig.add_subplot(515)

    res_h = res['Hartmann6 random subspace, D=1000']

    for idx, m in enumerate(plot_method_names):
        res_name = method_idx_to_res_name[idx]
        if res_name not in res_h.objective_at_true_best:
            continue  # Not run on this problem
        Y = np.log(res_h.objective_at_true_best[res_name] - (-3.32237))
        f = Y.mean(axis=0)
        sem = Y.std(axis=0) / np.sqrt(Y.shape[0])
        x = np.arange(1, 201)
        color = plot_colors[m]
        ax5.plot(x, f, color=color, label=m)
        ax5.errorbar(x[9::10], f[9::10], yerr=2 * sem[9::10], color=color, alpha=0.5, ls='')

    ax5.set_xlim([0, 201])
    ax5.set_ylim([-2.1, 1.7])
    ax5.set_xlabel('Function evaluations', fontsize=9)
    ax5.set_ylabel('Log regret', fontsize=9)
    ax5.grid(alpha=0.2, zorder=-10)
    ax5.set_title(r'Hartmann6, $d$=6 random subspace, $D$=1000')

    plt.subplots_adjust(right=0.72, bottom=0.06, left=0.08, top=0.97, hspace=0.45)
    plt.savefig('pdfs/log_regrets.pdf', pad_inches=0)


if __name__ == '__main__':
    make_fig_S7()
