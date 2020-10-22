# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import json
import numpy as np

from ax.storage.json_store.decoder import object_from_json

from plot_config import *


def extract_ablation_results():
    with open(f'../benchmarks/results/ablation_aggregated_results.json', 'r') as fin:
        res = object_from_json(json.load(fin))

    # Results
    ys = {
        'ALEBO': res['Branin, D=100_ablation'].objective_at_true_best['ALEBO, base'],
        'Ablation: Matern kernel': res['Branin, D=100_ablation'].objective_at_true_best['ALEBO, kernel ablation'],
        'Ablation: Normal projection': res['Branin, D=100_ablation'].objective_at_true_best['ALEBO, projection ablation'],
    }
    return ys


def make_fig_S_ablation():
    ys = extract_ablation_results()

    fig = plt.figure(figsize=(4, 2.5))
    ax = fig.add_subplot(111)

    x = np.arange(1, 51)
    for k, y in ys.items():
        f = y.mean(axis=0)
        sem = y.std(axis=0) / np.sqrt(y.shape[0])
        ax.errorbar(x, f, yerr=2 * sem, label=k)

    ax.set_ylim([0, 7])
    ax.set_yticks([0, 2, 4, 6])
    ax.legend(fontsize=7, loc='lower left')
    ax.set_title(r'Branin, $D=100$')
    ax.set_ylabel('Best value found', fontsize=9)
    ax.set_xlabel('Function evaluations', fontsize=9)
    ax.axhline(y=0.397887, c='gray', ls='--')
    ax.grid(alpha=0.2, zorder=-10)
    ax.set_xlim([0, 51])

    plt.subplots_adjust(right=0.995, bottom=0.16, left=0.1, top=0.91, wspace=0.05)
    plt.savefig('pdfs/branin_ablation_traces.pdf', pad_inches=0)


if __name__ == '__main__':
    make_fig_S_ablation()
