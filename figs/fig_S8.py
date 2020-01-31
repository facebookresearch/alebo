# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import json
import numpy as np

from ax.storage.json_store.decoder import object_from_json

from plot_config import *


def extract_sensitivity_results():
    res = {}
    for fname in [
        'branin_gramacy_100',
        'sensitivity',
    ]:
        with open(f'../benchmarks/results/{fname}_aggregated_results.json', 'r') as fin:
            res.update(object_from_json(json.load(fin)))

    # Results for D=100
    ys1 = {}
    for d in [2, 3, 4, 5, 6, 7, 8]:
        if d == 4:
            ys1[d] = res['Branin, D=100'].objective_at_true_best['ALEBO']
        else:
            ys1[d] = res['Branin, D=100_sensitivity'].objective_at_true_best[f'ALEBO, d={d}']

    # Results for d_e=4
    ys2 = {}
    for D in [50, 100, 200, 500, 1000]:
        if D == 100:
            ys2[D] = res['Branin, D=100'].objective_at_true_best['ALEBO']
        else:
            ys2[D] = res[f'Branin, D={D}_sensitivity'].objective_at_true_best['ALEBO']
    return ys1, ys2


def make_fig_S8():
    ys1, ys2 = extract_sensitivity_results()

    fig = plt.figure(figsize=(5.5, 2.2))
    ax = fig.add_subplot(121)

    x = np.arange(1, 51)
    for d_e in [2, 3, 4, 6, 8]:
        ax.plot(x, ys1[d_e].mean(axis=0), label=f'$d_e={d_e}$')

    ax.set_ylim([0, 7])
    ax.set_yticks([0, 2, 4, 6])
    ax.legend(fontsize=7)
    ax.set_title(r'Branin, $D=100$')
    ax.set_ylabel('Best value found', fontsize=9)
    ax.set_xlabel('Function evaluations', fontsize=9)
    ax.axhline(y=0.397887, c='gray', ls='--')
    ax.grid(alpha=0.2, zorder=-10)
    ax.set_xlim([0, 51])

    ax = fig.add_subplot(122)

    for D in [50, 100, 200, 500, 1000]:
        ax.plot(x, ys2[D].mean(axis=0), label=f'$D={D}$')

    ax.set_title(r'Branin, $d_e=4$')

    ax.set_ylim([0, 7])
    ax.legend(fontsize=7)
    ax.set_xlabel('Function evaluations', fontsize=9)
    ax.axhline(y=0.397887, c='gray', ls='--')
    ax.grid(alpha=0.2, zorder=-10)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_xlim([0, 51])
    ax.set_yticklabels([])

    plt.subplots_adjust(right=0.995, bottom=0.16, left=0.06, top=0.91, wspace=0.05)
    plt.savefig('pdfs/branin_by_d_D_traces.pdf', pad_inches=0)


if __name__ == '__main__':
    make_fig_S8()
