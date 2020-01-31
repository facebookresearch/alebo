# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

from fig_S8 import *


def make_fig_S9():
    ys1, ys2 = extract_sensitivity_results()

    d_es = [2, 3, 4, 5, 6, 7, 8]
    mus_de = []
    sems_de = []
    for d_e in d_es:
        Y = ys1[d_e][:, 49]
        mus_de.append(Y.mean())
        sems_de.append(Y.std() / np.sqrt(len(Y)))

    Ds = [50, 100, 200, 500, 1000]
    mus_D = []
    sems_D = []
    for D in Ds:
        Y = ys2[D][:, 49]
        mus_D.append(Y.mean())
        sems_D.append(Y.std() / np.sqrt(len(Y)))

    fig = plt.figure(figsize=(5.5, 1.8))

    ax = fig.add_subplot(121)
    ax.errorbar(d_es, mus_de, yerr=2*np.array(sems_de))
    ax.set_ylim([0.3, 2.85])
    ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel(r'Embedding dimension $d_e$', fontsize=9)
    ax.set_title(r'Branin, $D=100$')
    ax.set_ylabel('Best value found', fontsize=9)
    ax.grid(alpha=0.2)

    ax = fig.add_subplot(122)
    ax.errorbar(Ds, mus_D, yerr=2*np.array(sems_D))
    ax.set_ylim([0.3, 2.85])
    ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_yticklabels([])
    ax.set_xlabel(r'Ambient dimension $D$', fontsize=9)
    ax.set_title('Branin, $d_e=4$')
    ax.grid(alpha=0.2)
    ax.set_xticks([50, 200, 500, 1000])

    plt.subplots_adjust(right=0.93, bottom=0.19, left=0.12, top=0.89, wspace=0.1)
    plt.savefig('pdfs/branin_by_D_d.pdf', pad_inches=0)


if __name__ == '__main__':
    make_fig_S9()
