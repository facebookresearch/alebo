# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import pickle

from fig_1 import *


def evaluate_function_in_polytope():
    # Generate projection matrix
    np.random.seed(3)
    B0 = np.random.randn(2, 100)  # A REMBO projection
    B = B0 / np.sqrt((B0 ** 2).sum(axis=0))  # A hypersphere projection

    # Generate grid in low-d space
    b = 60.  # Something big to be sure we capture the whole range
    density = 1000
    grid_x = np.linspace(-b, b, density)
    grid_y = np.linspace(-b, b, density)
    grid2_x, grid2_y = np.meshgrid(grid_x, grid_y)
    X = np.array([grid2_x.flatten(), grid2_y.flatten()]).transpose()

    # Project up
    Y = (np.linalg.pinv(B) @ X.transpose()).transpose()
    z = ((Y<-1).any(axis=1) | (Y>1).any(axis=1))  # Points outside box bounds
    Y[z, :] = 0.  # Set them to 0 for now; we'll drop them later
    fs = branin_centered(Y[:, :2])
    # Drop points that violate polytope constraints in (1)
    fs[z] = np.nan
    fs = fs.reshape(grid2_x.shape)

    # Same thing with B0 instead of B.
    Y = (np.linalg.pinv(B0) @ X.transpose()).transpose()
    z = ((Y<-1).any(axis=1) | (Y>1).any(axis=1))  # Points outside box bounds
    Y[z, :] = 0.  # Set them to 0 for now; we'll drop them later
    fs_B0 = branin_centered(Y[:, :2])
    fs_B0[z] = np.nan  # Drop points outside the box bounds
    fs_B0 = fs_B0.reshape(grid2_x.shape)
    with open('data/figS4_sim_output.pckl', 'wb') as fout:
        pickle.dump((grid_x, grid_y, fs, fs_B0), fout)


def make_fig_S4():
    with open('data/figS4_sim_output.pckl', 'rb') as fin:
        grid_x, grid_y, fs, fs_B0 = pickle.load(fin)

    fig = plt.figure(figsize=(5.5, 2))
    plt.set_cmap('RdBu_r')

    ax = fig.add_subplot(121)
    CS1 = ax.contourf(grid_x, grid_y, np.log(fs_B0), levels=np.linspace(-1, 6, 30))
    ax.grid(False)
    ax.set_xlabel(r'$x_1$', fontsize=9)
    ax.set_ylabel(r'$x_2$', fontsize=9)
    ax.set_xlim([-45, 45])
    ax.set_ylim([-35, 35])

    ax = fig.add_subplot(122)
    CS1 = ax.contourf(grid_x, grid_y, np.log(fs), levels=np.linspace(-1, 6, 30))
    ax.grid(False)
    ax.set_xlabel(r'$x_1$', fontsize=9)
    ax.set_ylabel(r'$x_2$', fontsize=9)
    ax.set_xlim([-62, 62])
    ax.set_ylim([-50, 50])

    plt.subplots_adjust(right=0.99, top=0.99, left=0.1, bottom=0.17, wspace=0.45)
    plt.savefig('pdfs/new_embedding.pdf', pad_inches=0)


if __name__ == '__main__':
    #evaluate_function_in_polytope()  # Takes 20s to run, creates data/figS4_sim_output.pckl
    make_fig_S4()
