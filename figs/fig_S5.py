# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import numpy as np
import pickle

from plot_config import *


def make_fig_S5():
    with open('data/fig4_sim_output.pckl', 'rb') as fin:
        res = pickle.load(fin)

    nsamp = 1000
    fig = plt.figure(figsize=(5.5, 2))
    for i, d in enumerate([2, 6, 10]):
        ax = fig.add_subplot(1, 3, i + 1)
        x = [d_use for d_use in range(21) if d_use >= d]
        y1 = np.array([res['rembo'][(100, d, d_use)] for d_use in x])
        y2 = np.array([res['hesbo'][(100, d, d_use)] for d_use in x])
        y3 = np.array([res['unitsphere'][(100, d, d_use)] for d_use in x])
        y1err = 2 * np.sqrt(y1 * (1 - y1) / nsamp)
        y2err = 2 * np.sqrt(y2 * (1 - y2) / nsamp)
        y3err = 2 * np.sqrt(y3 * (1 - y3) / nsamp)
        ax.errorbar(x, y1, yerr=y1err, color=plt.cm.tab10(0), marker='')
        ax.errorbar(x, y2, yerr=y2err, color=plt.cm.tab10(1), marker='')
        ax.errorbar(x, y3, yerr=y3err, color=plt.cm.tab10(2), marker='')
        ax.set_title(r'$d={d}$'.format(d=d))
        if i == 0:
            ax.set_ylabel('Probability embedding\ncontains optimizer', fontsize=9)
            ax.legend(['REMBO', 'HeSBO', r'Hypersphere'], loc='lower right', fontsize=7)
        ax.set_xlabel(r'$d_e$', fontsize=9)
        ax.set_xlim([0, 21])
        ax.set_ylim([-0.02, 1.02])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if i > 0:
            ax.set_yticklabels([])
        ax.grid(True, alpha=0.2)

    plt.subplots_adjust(right=0.99, bottom=0.17, left=0.10, top=0.89, wspace=0.1)

    plt.savefig('pdfs/lp_solns_ext.pdf', pad_inches=0)

if __name__ == '__main__':
    # Assumes fig_4 has been run
    make_fig_S5()
