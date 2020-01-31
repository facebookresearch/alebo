# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import numpy as np

from fig_1 import *


def hesbo_branin(X, mode):
    # In 2-D, HESBO has 3 possible embeddings:
    # Full rank, span x1 and x2
    # x1=x2
    # x1=-x2
    Y = X.copy()
    if mode == 1:
        pass  # All is well here!
    elif mode == 2:
        Y[:, 0] = Y[:, 1]
    elif mode == 3:
        Y[:, 0] = -Y[:, 1]
    return branin_centered(Y)


def make_fig_S1():
    # Evaluate the branin function on the grid under the three possible embeddings
    grid_xhes, grid_yhes, fs_hesbo1 = eval_f_on_grid(hesbo_branin, [-1, 1], [-1, 1], {'mode': 1}, 2, density=1000)
    grid_xhes, grid_yhes, fs_hesbo2 = eval_f_on_grid(hesbo_branin, [-1, 1], [-1, 1], {'mode': 3}, 2, density=1000)
    grid_xhes, grid_yhes, fs_hesbo3 = eval_f_on_grid(hesbo_branin, [-1, 1], [-1, 1], {'mode': 2}, 2, density=1000)

    fig = plt.figure(figsize=(5.5, 1.8), facecolor='w', edgecolor='w')
    plt.set_cmap('RdBu_r')

    for i, fs in enumerate([fs_hesbo1, fs_hesbo2, fs_hesbo3]):
        ax = fig.add_subplot(1, 3, i + 1)
        CS1 = ax.contourf(grid_xhes, grid_yhes, np.log(fs), levels=np.linspace(-1, 6, 30))
        ax.grid(False)
        ax.set_xlabel(r'$x_1$', fontsize=9)
        ax.set_xlim([-1, 1])
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        if i == 0:
            ax.set_ylabel(r'$x_2$', fontsize=9)
            ax.set_ylim([-1, 1])
        else:
            ax.set_yticklabels([])

    plt.subplots_adjust(right=0.98, top=0.975, left=0.1, bottom=0.195)
    plt.savefig('pdfs/hesbo_embeddings.pdf', pad_inches=0)


if __name__ == '__main__':
    make_fig_S1()
