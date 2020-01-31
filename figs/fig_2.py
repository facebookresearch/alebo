# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import numpy as np

from plot_config import *


def make_fig_2():
    # Run the simulation
    np.random.seed(1)

    Ds = [20, 100, 1000]
    ds = list(range(1, 6))
    nsamp = 1000
    p_interior = {}
    for D in Ds:
        for d in ds:
            p_interior[(D, d)] = 0.
            for _ in range(nsamp):
                # Generate a REMBO projection
                A = np.random.randn(D, d)
                # Sample a point in [-sqrt(d), sqrt(d)]^d
                x = (np.random.rand(d) * 2 - 1) * np.sqrt(d)
                # Project up
                z = A @ x
                # Check if satisfies box bounds
                if z.min() >= -1 and z.max() <= 1:
                    p_interior[(D, d)] += 1
            p_interior[(D, d)] /= nsamp

    # Make the figure
    fig = plt.figure(figsize=(2.5, 1.5))
    ax = fig.add_subplot(111)

    for i, D in enumerate(Ds):
        ax.plot(ds, [p_interior[(D, d)] for d in ds], 'x-', c=plt.cm.tab10(i))
    ax.legend([r'$D=20$', r'$D=100$', r'$D=1000$'], fontsize=7)
    ax.set_xlabel(r'Embedding dimension $d_e$', fontsize=9)
    ax.set_ylabel('Probability projection\nsatisfies box bounds', fontsize=9)
    plt.subplots_adjust(right=0.99, bottom=0.23, left=0.22, top=0.94)

    plt.savefig('pdfs/rembo_p_interior.pdf', pad_inches=0)

if __name__ == '__main__':
    make_fig_2()
