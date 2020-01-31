# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import numpy as np
import pickle

from plot_config import *


def make_fig_S2():
    # Load in simulation results
    with open('data/fig3_sim_output.pckl', 'rb') as fin:
        (test_Y, f1, var1, f2, var2, f3, var3) = pickle.load(fin)

    fig = plt.figure(figsize=(5.5, 2))

    ax = fig.add_subplot(131)
    ax.errorbar(
        x=test_Y.numpy(), y=f3.numpy(), yerr = 2 * np.sqrt(var3.numpy()),
        c='gray', lw=1, ls='', marker='.', mfc='k', mec='k', ms=3
    )
    x0 = -2.5
    x1 = 0.5
    ax.plot([x0, x1], [x0, x1],  '-', zorder=-5, alpha=0.5, c='steelblue', lw=2)
    ax.set_xlim([x0, x1])
    ax.set_ylim([x0, x1])
    ax.set_xlabel('True value', fontsize=9)
    ax.set_ylabel('Model prediction', fontsize=9)
    ax.set_title('ARD RBF', fontsize=9)


    ax = fig.add_subplot(132)
    ax.errorbar(
        x=test_Y.numpy(), y=f2.numpy(), yerr = 2 * np.sqrt(var2.numpy()),
        c='gray', lw=1, ls='', marker='.', mfc='k', mec='k', ms=3
    )
    x0 = -2.5
    x1 = 0.5
    ax.plot([x0, x1], [x0, x1],  '-', zorder=-5, alpha=0.5, c='steelblue', lw=2)
    ax.set_xlim([x0, x1])
    ax.set_ylim([x0, x1])
    ax.set_xlabel('True value', fontsize=9)
    ax.set_title('Mahalanobis\npoint estimate', fontsize=9)


    ax = fig.add_subplot(133)
    ax.errorbar(
        x=test_Y.numpy(), y=f1.numpy(), yerr = 2 * np.sqrt(var1.numpy()),
        c='gray', lw=1, ls='', marker='.', mfc='k', mec='k', ms=3
    )
    x0 = -3.
    x1 = 1
    ax.plot([x0, x1], [x0, x1],  '-', zorder=-5, alpha=0.5, c='steelblue', lw=2)
    ax.set_xlim([x0, x1])
    ax.set_ylim([x0, x1])
    ax.set_title('Mahalanobis\nposterior sampled', fontsize=9)
    ax.set_xticks([-3, -2, -1, 0, 1])
    ax.set_xlabel('True value', fontsize=9)

    plt.subplots_adjust(right=0.99, bottom=0.17, left=0.1, top=0.84, wspace=0.3)
    plt.savefig('pdfs/model_predictions.pdf', pad_inches=0)


if __name__ == '__main__':
    # Assumes fig_3 has already been run
    make_fig_S2()
