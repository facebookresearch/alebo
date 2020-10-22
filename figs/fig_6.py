import json
import numpy as np

from ax.storage.json_store.decoder import object_from_json

from plot_config import *


def make_nasbench_figure():
    with open('../benchmarks/results/nasbench_aggregated_results.json', 'r') as fin:
        res = object_from_json(json.load(fin))

    # A map from method idx in plot_method_names to the name used in res
    method_idx_to_res_name = {
        1: 'REMBO',
        3: 'HeSBO',
        9: 'cmaes',
        10: 'turbo',
        11: 'Sobol',
        0: 'ALEBO',
    }
    plot_method_names[3] = 'HeSBO'
    plot_colors['HeSBO'] = plt.cm.tab20(3)

    fig = plt.figure(figsize=(2.96, 2.0))
    ax1 = fig.add_subplot(111)
    method_idx = 1
    method_names_used = []
    for i, m in method_idx_to_res_name.items():
        f = np.nanmean(res[m], axis=0)
        sem = np.nanstd(res[m], axis=0) / np.sqrt(res[m].shape[0])
        x = np.arange(1, 51)
        mname = plot_method_names[i]
        color = plot_colors[mname]
        ax1.plot(x, f, color=color, label=mname)
        ax1.errorbar(x, f, yerr=2 * sem, color=color, alpha=0.5, ls='')

    ax1.set_xlim([0, 51])
    ax1.set_ylabel('Best feasible test accuracy', fontsize=9)
    ax1.set_xlabel('Function evaluations', fontsize=9)

    #ax1.legend(bbox_to_anchor=(1.0, 1.24), ncol=4, fontsize=6, columnspacing=1.65)
    ax1.legend(ncol=2, loc='lower right', fontsize=7)

    ax1.set_ylim([0.92, 0.936])
    ax1.set_yticks([0.92, 0.925, 0.93, 0.935])
    ax1.set_yticklabels(['92.0\%', '92.5\%', '93.0\%', '93.5\%'])
    ax1.grid(alpha=0.2, zorder=-10)

    plt.subplots_adjust(right=0.98, bottom=0.17, left=0.19, top=0.98)
    plt.savefig(f'pdfs/nas.pdf', pad_inches=0)
    #plt.show()


if __name__ == '__main__':
    make_nasbench_figure()
