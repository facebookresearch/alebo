# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

from fig_4 import *


def run_simulation():
    t1 = time.time()
    nsamp = 1000
    res = {'unitsphere': {}}
    for D in [50, 100, 200]:
        for d in range(2, 19, 2):
            for d_use in range(2, 21, 2):
                if d_use < d:
                    continue
                res['unitsphere'][(D, d, d_use)] = p_A_contains_optimizer(
                    d=d, D=D, d_use=d_use, gen_A_fn=gen_A_unitsphere, nsamp=nsamp
                )
    with open('data/figS6_sim_output.pckl', 'wb') as fout:
        pickle.dump(res, fout)
    print(time.time() - t1)


def make_fig_S6():
    with open('data/figS6_sim_output.pckl', 'rb') as fin:
        res = pickle.load(fin)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4,figsize=(5.5, 2), 
                    gridspec_kw={"width_ratios":[1, 1, 1, 0.15]})

    axes = [ax1, ax2, ax3]

    for i, D in enumerate([50, 100, 200]):
        
        ds = []
        duses = []
        ps = []
        for d in range(2, 19, 2):
            for d_use in range(2, 21, 2):
                if d_use < d:
                    continue
                ds.append(d)
                duses.append(d_use)
                ps.append(res['unitsphere'][(D, d, d_use)])

        cntr = axes[i].tricontourf(duses, ds, ps, levels=np.linspace(0, 1.001, 21), cmap='viridis')
        axes[i].set_title(f'$D={D}$')
        if i == 0:
            axes[i].set_yticks([2, 6, 10, 14, 18])
            axes[i].set_ylabel(r'True subspace dimension $d$', fontsize=9)
        else:
            axes[i].set_yticks([2, 6, 10, 14, 18])
            axes[i].set_yticklabels([])
        axes[i].grid(alpha=0.2, zorder=-2)
        axes[i].set_xlabel(r'Embedding $d_e$')
        axes[i].set_xticks([2, 6, 10, 14, 18])

    ax4.patch.set_visible(False)
    ax4.set_yticks([])
    ax4.set_xticks([])
    fig.colorbar(cntr, ax=ax4, ticks=[0, 0.25, 0.5, 0.75, 1.], fraction=1)

    plt.subplots_adjust(right=0.97, bottom=0.2, left=0.07, top=0.89, wspace=0.1)
    plt.savefig('pdfs/lp_solns_D.pdf', pad_inches=0)


if __name__ == '__main__':
    #run_simulation()  # This takes about 3hrs and produces data/figS6_sim_output.pckl
    make_fig_S6()
