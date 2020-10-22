# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import json
import numpy as np

from ax.storage.json_store.decoder import object_from_json

from plot_config_nr import *


def make_fig_5():
    # Load in the benchmark results
    res = {}
    for fname in ['hartmann6_1000', 'branin_gramacy_100']:
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
    fig = plt.figure(figsize=(5.5, 3.7))

    ####### Branin, D=100
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(234)

    res_h = res['Branin, D=100']

    for idx, m in enumerate(plot_method_names):
        res_name = method_idx_to_res_name[idx]
        if res_name not in res_h.objective_at_true_best:
            continue  # Not run on this problem
        Y = res_h.objective_at_true_best[res_name]
        f = Y.mean(axis=0)
        x = np.arange(1, 51)
        color = plot_colors[m]
        ax1.plot(x, f, color=color, label=m)

        parts = ax2.violinplot(positions=[idx], dataset=Y[:, 49], showmeans=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
        for field in ['cmeans', 'cmaxes', 'cmins', 'cbars']:
            parts[field].set_color(color)

    ax1.set_xlim([0, 51])
    ax1.set_ylabel('Best value found', fontsize=7)
    ax1.set_xlabel('Function evaluations', fontsize=7)

    ax1.axhline(y=0.397887, c='gray', ls='--')
    ax1.grid(alpha=0.2, zorder=-10)
    ax1.set_ylim([0, 7])


    ax2.set_xticks(range(12))
    ax2.set_xticklabels([])
    ax2.set_ylabel('Final value', fontsize=7)
    ax2.grid(alpha=0.2, zorder=-10)
    ax2.set_xticklabels([plot_method_names[i] for i in range(12)], fontsize=6)
    ax2.xaxis.set_tick_params(rotation=90)

    
    # Make the legend
    custom_lines = []
    names = []
    for i in range(12):
        names.append(plot_method_names[i])
        custom_lines.append(
            Line2D([0], [0], color=plot_colors[plot_method_names[i]], lw=2)
        )

    order = range(12)
    names = [names[o] for o in order]
    custom_lines = [custom_lines[o] for o in order]
    ax1.legend(custom_lines, names, ncol=6, fontsize=5.5, bbox_to_anchor=(3.52, -2.26))
    ax1.set_title('Branin, $d$=2, $D$=100', fontsize=8)

    ####### Hartmann6, D=1000
    ax1 = fig.add_subplot(232)
    ax2 = fig.add_subplot(235)

    res_h = res['Hartmann6, D=1000']

    for idx, m in enumerate(plot_method_names):
        res_name = method_idx_to_res_name[idx]
        if res_name not in res_h.objective_at_true_best:
            continue  # Not run on this problem
        Y = res_h.objective_at_true_best[res_name]
        f = Y.mean(axis=0)
        x = np.arange(1, 201)
        color = plot_colors[m]
        ax1.plot(x, f, color=color, label=m)

        parts = ax2.violinplot(positions=[idx], dataset=Y[:, 199], showmeans=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
        for field in ['cmeans', 'cmaxes', 'cmins', 'cbars']:
            parts[field].set_color(color)

    ax1.set_xlim([0, 201])
    #ax1.set_ylabel('Best value found', fontsize=9)
    ax1.set_xlabel('Function evaluations', fontsize=7)

    ax1.axhline(y=-3.32237, c='gray', ls='--')
    ax1.grid(alpha=0.2, zorder=-10)
    ax1.set_ylim([-3.5, -0.5])

    ax2.set_xticks(range(12))
    ax2.set_xticklabels([])
    #ax2.set_ylabel('Final value', fontsize=9)
    ax2.grid(alpha=0.2, zorder=-10)
    ax2.set_xticklabels([plot_method_names[i] for i in range(12)], fontsize=6)
    ax2.xaxis.set_tick_params(rotation=90)
    ax1.set_title('Hartmann6, $d$=6, $D$=1000', fontsize=8)

    ####### Gramacy, D=100
    ax1 = fig.add_subplot(233)
    ax2 = fig.add_subplot(236)

    res_h = res['Gramacy, D=100']

    for idx, m in enumerate(plot_method_names):
        res_name = method_idx_to_res_name[idx]
        if res_name not in res_h.objective_at_true_best:
            continue  # Not run on this problem
        Y = res_h.objective_at_true_best[res_name]
        f = Y.mean(axis=0)
        x = np.arange(1, 51)
        color = plot_colors[m]
        ax1.plot(x, f, color=color, label=m)

        parts = ax2.violinplot(positions=[idx], dataset=Y[:, 49], showmeans=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
        for field in ['cmeans', 'cmaxes', 'cmins', 'cbars']:
            parts[field].set_color(color)
    
    ax1.set_xlim([0, 51])
    #ax1.set_ylabel('Best value found', fontsize=9)
    ax1.set_xlabel('Function evaluations', fontsize=7)
    ax1.set_ylim([0.58, 1])

    ax1.axhline(y=0.5998, c='gray', ls='--')
    ax1.grid(alpha=0.2, zorder=-10)

    ax2.set_xticks(range(12))
    ax2.set_xticklabels([plot_method_names[i] for i in range(12)], fontsize=6)
    ax2.xaxis.set_tick_params(rotation=90)
    #ax2.set_ylabel('Final value', fontsize=9)
    ax2.grid(alpha=0.2, zorder=-10)
    ax1.set_title('Gramacy, $d$=2, $D$=100', fontsize=8)

    plt.subplots_adjust(right=0.995, bottom=0.3, left=0.07, top=0.94, wspace=0.25, hspace=0.45)
    plt.savefig('pdfs/benchmark_results_t.pdf', pad_inches=0)


if __name__ == '__main__':
    make_fig_5()
