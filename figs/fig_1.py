# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import numpy as np

from ax.utils.measurement.synthetic_functions import branin, hartmann6

from plot_config import *


def branin_centered(X):
    # Evaluate branin problem, scaled to X \in [-1, 1]^2
    # Map from [-1, 1]^2 to [[-5, 10], [0, 15]]
    assert X.min() >= -1
    assert X.max() <= 1
    Xu = (X + 1) / 2.
    Xu *= 15
    Xu[:, 0] -= 5
    return branin(Xu)

def hartmann6_centered(X):
    # Evaluate hartmann6 problem, scaled to X \in [-1, 1]^2
    # Map from [-1, 1]^6 to [0, 1]^6
    assert X.min() >= -1
    assert X.max() <= 1
    Xu = (X + 1) / 2.
    return hartmann6(Xu)

def rembo_branin(X, A):
    # Map from low-d to high-D
    Y = (A @ X.transpose()).transpose()
    # Clip to [-1, 1]
    Y = np.clip(Y, a_min=-1, a_max=1)
    # Evaluate Branin on first two components
    return branin_centered(Y[:, :2])

def rembo_hartmann6(X, A):
    # Map from low-d to high-D
    Y = (A @ X.transpose()).transpose()
    # Clip to [-1, 1]
    Y = np.clip(Y, a_min=-1, a_max=1)
    # Evaluate Hartmann6 on first six components
    return hartmann6_centered(Y[:, :6])

def eval_f_on_grid(f, bounds_x, bounds_y, f_kwargs, d, density=100):
    # prepare the grid on which to evaluate the problem
    grid_x = np.linspace(bounds_x[0], bounds_x[1], density)
    grid_y = np.linspace(bounds_y[0], bounds_y[1], density)
    grid2_x, grid2_y = np.meshgrid(grid_x, grid_y)
    X = np.array([grid2_x.flatten(), grid2_y.flatten()]).transpose()
    if d > 2:
        # Add in the other components, just at 0
        X = np.hstack((X, np.zeros((X.shape[0], d - 2))))
    fs = f(X, **f_kwargs).reshape(grid2_x.shape)
    return grid_x, grid_y, fs

def make_fig_1():
    ## Branin
    # Evaluate the usual Branin problem, but scaled to [-1, 1]^2
    grid_x1, grid_y1, fs_branin = eval_f_on_grid(branin_centered, [-1, 1], [-1, 1], {}, 2)

    # Generate a REMBO projection matrix
    D = 100
    np.random.seed(1)
    A_b = np.random.randn(D, 2)
    # Evaluate the function across the low-d space
    bounds = [-np.sqrt(2), np.sqrt(2)]
    grid_x2, grid_y2, fs_rembo = eval_f_on_grid(rembo_branin, bounds, bounds, {'A': A_b}, 2)

    ## Hartmann6
    # Evaluate the usual Hartmann6 problem, but scaled to [-1, 1]^6
    grid_x1h, grid_y1h, fs_hartmann6 = eval_f_on_grid(hartmann6_centered, [-1, 1], [-1, 1], {}, 6)

    # Generate a REMBO projection matrix
    D = 100
    A_h = np.random.randn(D, 6)
    # Evaluate the function across the low-d space
    bounds = [-np.sqrt(6), np.sqrt(6)]
    grid_x2h, grid_y2h, fs_rembo_h = eval_f_on_grid(rembo_hartmann6, bounds, bounds, {'A': A_h}, 6)

    # Make the figure
    fig = plt.figure(figsize=(5.5, 1.2), facecolor='w', edgecolor='w')
    plt.set_cmap('RdBu_r')

    ### Branin

    ax = fig.add_subplot(141)
    CS1 = ax.contourf(grid_x1, grid_y1, np.log(fs_branin), levels=np.linspace(-1, 6, 30))
    ax.grid(False)
    #ax.set_xlabel(r'$x_1$', fontsize=9)
    ax.set_xlim([-1, 1])
    #ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticks([])
    #ax.set_ylabel(r'$x_2$', fontsize=9)
    ax.set_ylim([-1, 1])
    ax.set_yticks([])
    #ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_title(r'Branin function, $d$=2')

    ax = fig.add_subplot(142)
    CS1 = ax.contourf(grid_x2, grid_y2, np.log(fs_rembo), levels=np.linspace(-1, 6, 30))
    ax.grid(False)
    #ax.set_xlabel(r'$x_1$', fontsize=9)
    ax.set_xlim([-np.sqrt(2), np.sqrt(2)])
    ax.set_xticks([])
    #ax.set_xticks([-1.4, -1, -0.5, 0, 0.5, 1, 1.4])
    #ax.set_ylabel(r'$x_2$', fontsize=9)
    ax.set_ylim([-np.sqrt(2), np.sqrt(2)])
    ax.set_yticks([])
    #ax.set_yticks([-1.4, -1, -0.5, 0, 0.5, 1, 1.4])
    ax.set_title('REMBO embedding,\n$D$=100, $d_e$=2')

    ### Hartmann6

    ax = fig.add_subplot(143)
    CS1f = ax.contourf(grid_x1h, grid_y1h, fs_hartmann6, levels=np.linspace(-1.2, 0., 20))
    ax.grid(False)
    #ax.set_xlabel(r'$x_1$', fontsize=9)
    ax.set_xlim([-1, 1])
    ax.set_xticks([])
    #ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    #ax.set_ylabel(r'$x_2$', fontsize=9)
    ax.set_ylim([-1, 1])
    ax.set_yticks([])
    #ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_title(r'Hartmann6 function, $d$=6')

    ax = fig.add_subplot(144)
    CS1f = ax.contourf(grid_x2h, grid_y2h, fs_rembo_h, levels=np.linspace(-1.2, 0., 20))
    ax.grid(False)
    #ax.set_xlabel(r'$x_1$', fontsize=9)
    ax.set_xlim([-np.sqrt(6), np.sqrt(6)])
    ax.set_xticks([])
    #ax.set_xticks([-2, -1, 0, 1, 2,])
    #ax.set_ylabel(r'$x_2$', fontsize=9)
    ax.set_ylim([-np.sqrt(6), np.sqrt(6)])
    ax.set_yticks([])
    #ax.set_yticks([-2, -1, 0, 1, 2,])
    ax.set_title('REMBO embedding,\n$D$=100, $d_e$=6')

    fig.subplots_adjust(wspace=0.13, top=0.74, bottom=0.05, right=0.99, left=0.01)
    plt.savefig('pdfs/rembo_illustrations_w.pdf', pad_inches=0)

if __name__ == '__main__':
    make_fig_1()
