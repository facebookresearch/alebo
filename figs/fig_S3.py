# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

from math import pi

from fig_3 import *


def compute_ll(f, var, test_Y):
    return -0.5 * (torch.log(2 * pi * var) + ((test_Y - f) ** 2) / var).sum().item()


def run_simulation():
    D = 100
    d = 6

    # Get projection
    torch.manual_seed(10)
    B0 = torch.randn(d, D, dtype=torch.double)
    B = B0 / torch.sqrt((B0 ** 2).sum(dim=0))

    # Get test data
    _, _, _, test_X, test_Y, _, _ = gen_train_test_sets(B, ntrain=10, ntest=1000, seed_test=1000)

    ns = np.array([40, 50, 75, 100, 125, 150, 175, 200])
    nrep = 20

    ll_alebo = np.zeros((nrep, len(ns)))
    ll_ard = np.zeros((nrep, len(ns)))

    for i in range(nrep):    
        for j, n in enumerate(ns):
            # Generate training data
            train_X, train_Y, train_Yvar, _, _, mu, sigma = gen_train_test_sets(
                B, ntrain=n, ntest=10, seed_train=(i + 1) * len(ns) + j
            )

            # Predict with each model
            f1, var1 = fit_and_predict_alebo(B, train_X, train_Y, train_Yvar, test_X, mu, sigma)
            f3, var3 = fit_and_predict_ARDRBF(B, train_X, train_Y, train_Yvar, test_X, mu, sigma)
            ll_alebo[i, j] = compute_ll(f1, var1, test_Y)
            ll_ard[i, j] = compute_ll(f3, var3, test_Y)

    # Save outcome
    with open('data/figS3_sim_output.pckl', 'wb') as fout:
        pickle.dump((ns, ll_alebo, ll_ard), fout)


def make_fig_S3():
    with open('data/figS3_sim_output.pckl', 'rb') as fin:
        (ns, ll_alebo, ll_ard) = pickle.load(fin)

    ntest = 1000.
    nrep = 20

    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(111)
    ax.errorbar(ns, ll_alebo.mean(axis=0) / ntest, yerr=2 * ll_alebo.std(axis=0)/ ntest / np.sqrt(nrep))
    ax.errorbar(ns, ll_ard.mean(axis=0) / ntest, yerr=2 * ll_ard.std(axis=0)/ ntest / np.sqrt(nrep))
    ax.grid(alpha=0.2)
    ax.set_ylabel('Average test-set\nlog likelihood', fontsize=9)
    ax.set_xlabel('Training set size', fontsize=9)
    ax.legend(['Mahalanobis, sampled', 'ARD RBF'], fontsize=7, loc='lower right')

    plt.subplots_adjust(right=0.995, bottom=0.19, left=0.21, top=0.98, wspace=0.3)
    plt.savefig('pdfs/log_likelihood.pdf', pad_inches=0)

if __name__ == '__main__':
    run_simulation()  # Produces data/figS3_sim_output.pckl
    make_fig_S3()
