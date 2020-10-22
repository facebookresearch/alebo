# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import pickle
import time
import torch
import numpy as np
import cvxpy as cp  # 1.0.25

from plot_config import *


def gen_A_rembo(d: int, D: int) -> np.ndarray:
    A = torch.randn(D, d, dtype=torch.double)
    return A.numpy()

def gen_A_hesbo(d: int, D:int) -> np.ndarray:
    A = torch.zeros((D, d), dtype=torch.double)
    h = torch.randint(d, size=(D,))
    s = 2 * torch.randint(2, size=(D,), dtype=torch.double) - 1
    for i in range(D):
        A[i, h[i]] = s[i]
    return A.numpy()

def gen_A_unitsphere(d: int, D: int) -> np.ndarray:
    A = np.random.randn(D, d)  # A REMBO projection _up_
    A = A / np.sqrt((A ** 2).sum(axis=1))[:, None]
    return A

def A_contains_xstar(xstar, A, perm):
    d = len(xstar)
    D = A.shape[0]
    Acon = np.zeros((d, D))
    Acon[:d, :d] = np.diag(np.ones(d))
    # Shuffle columns, to place true embedding on columns perm
    Acon = Acon[:, perm]
    Q = A @ np.linalg.pinv(A) - np.eye(D)
    A_eq = np.vstack((Acon, Q))
    b_eq = np.hstack((xstar, np.zeros(D)))
    
    c = np.zeros(D)
    
    x = cp.Variable(D)
    prob = cp.Problem(
        cp.Minimize(c.T * x),
        [
            A_eq @ x == b_eq,
            x >= -1,
            x <= 1,
        ],
    )
    prob.solve(solver=cp.ECOS)
    
    if prob.status == cp.OPTIMAL:
        has_opt = True
    elif prob.status == cp.INFEASIBLE:
        has_opt = False
    else:
        raise ValueError(prob.status)
    return has_opt, prob

def p_A_contains_optimizer(d, D, d_use, gen_A_fn, nsamp):
    num_feas = 0.
    for _ in range(nsamp):
        # Sample location of optimizer uniformly on [-1, 1]^d
        xstar = np.random.rand(d) * 2 - 1
        # Sample features of embedding (first d) uniformly at random
        perm = list(range(D))
        np.random.shuffle(perm)
        # Generate projection matrix
        A = gen_A_fn(d_use, D)
        has_opt, _ = A_contains_xstar(xstar, A, perm)
        num_feas += float(has_opt)
    return num_feas / nsamp

def run_simulation1():
    t1 = time.time()
    nsamp = 1000
    res = {'rembo': {}, 'hesbo': {}, 'unitsphere': {}}
    D = 100
    for d in [2, 6, 10]:
        for d_use in range(1, 21):
            if d_use < d:
                continue
            res['rembo'][(D, d, d_use)] = p_A_contains_optimizer(
                d=d, D=D, d_use=d_use, gen_A_fn=gen_A_rembo, nsamp=nsamp
            )
            res['hesbo'][(D, d, d_use)] = p_A_contains_optimizer(
                d=d, D=D, d_use=d_use, gen_A_fn=gen_A_hesbo, nsamp=nsamp
            )
            res['unitsphere'][(D, d, d_use)] = p_A_contains_optimizer(
                d=d, D=D, d_use=d_use, gen_A_fn=gen_A_unitsphere, nsamp=nsamp
            )
    print(time.time() - t1)
    with open('data/fig4_sim_output.pckl', 'wb') as fout:
        pickle.dump(res, fout)

def make_fig_4():
    with open('data/fig4_sim_output.pckl', 'rb') as fin:
        res = pickle.load(fin)

    nsamp = 1000
    fig = plt.figure(figsize=(2.63, 1.45))
    for i, d in enumerate([2, 6]):
        ax = fig.add_subplot(1, 2, i + 1)
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
            ax.set_ylabel(r'$P_{\textrm{opt}}$', fontsize=9)
            ax.legend(['REMBO', 'HeSBO', r'Hypersphere'], loc='lower right', fontsize=5)
        ax.set_xlabel(r'$d_e$', fontsize=9)
        ax.set_xlim([0, 21])
        ax.set_ylim([-0.02, 1.02])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if i > 0:
            ax.set_yticklabels([])
        ax.grid(True, alpha=0.2)

    plt.subplots_adjust(right=0.99, bottom=0.23, left=0.17, top=0.87, wspace=0.1)

    plt.savefig('pdfs/lp_solns.pdf', pad_inches=0)

if __name__ == '__main__':
    #run_simulation1()  # Will take about 30mins, produces data/fig4_sim_output.pckl
    make_fig_4()
