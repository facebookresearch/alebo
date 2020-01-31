# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import time
import numpy as np
import torch
import pickle

from ax.models.torch.alebo import ALEBO
from ax.models.random.alebo_initializer import ALEBOInitializer
from ax.models.torch.botorch import BotorchModel

from botorch.test_functions.synthetic import Hartmann
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from torch import Tensor
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from gpytorch.distributions.multivariate_normal import MultivariateNormal

from plot_config import *


def highDhartmann6(X):
    # X \in [-1, 1]^D
    h = Hartmann()
    Xd = (X[:, :6] + 1) / 2.
    return h.evaluate_true(Xd)

def gen_train_test_sets(B, ntrain, ntest, seed_train=1000, seed_test=2000):
    # Generate training points
    m1 = ALEBOInitializer(B=B.numpy(), seed=seed_train)
    train_X = torch.tensor(m1.gen(n=ntrain, bounds=[])[0], dtype=torch.double)
    train_Y = highDhartmann6(train_X)
    # Standardize train Y
    mu = train_Y.mean()
    sigma = train_Y.std()
    train_Y = (train_Y - mu) / sigma
    train_Y = train_Y.unsqueeze(1)
    train_Yvar = 1e-7 * torch.ones(train_Y.shape)

    # Generate test points
    m2 = ALEBOInitializer(B=B.numpy(), seed=seed_test)
    test_X = torch.tensor(m2.gen(n=ntest, bounds=[])[0], dtype=torch.double)
    test_Y = highDhartmann6(test_X)
    return train_X, train_Y, train_Yvar, test_X, test_Y, mu, sigma

def fit_and_predict_alebo(B, train_X, train_Y, train_Yvar, test_X, mu, sigma):
    m = ALEBO(B=B)
    m.fit([train_X], [train_Y], [train_Yvar], [], [], [], [], [])
    f, var = m.predict(test_X)
    # Return predictions, un-standardized
    return f.squeeze() * sigma + mu, var.squeeze() * sigma ** 2

def fit_and_predict_map(B, train_X, train_Y, train_Yvar, test_X, mu, sigma):
    m = ALEBO(B=B, laplace_nsamp=1)  # laplace_nsamp=1 uses MAP estimate
    m.fit([train_X], [train_Y], [train_Yvar], [], [], [], [], [])
    f, var = m.predict(test_X)
    # Return predictions, un-standardized
    return f.squeeze() * sigma + mu, var.squeeze() * sigma ** 2

class ARDRBFGP(BatchedMultiOutputGPyTorchModel, ExactGP):
    """A GP with fixed observation noise and an ARD RBF kernel."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor) -> None:
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, train_Yvar = self._transform_tensor_args(
            X=train_X, Y=train_Y, Yvar=train_Yvar
        )
        likelihood = FixedNoiseGaussianLikelihood(
            noise=train_Yvar, batch_shape=self._aug_batch_shape
        )
        ExactGP.__init__(
            self, train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )
        self.mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(
                ard_num_dims=train_X.shape[-1],
                batch_shape=self._aug_batch_shape,
            ),
            batch_shape=self._aug_batch_shape,
        )
        self.to(train_X)
    
    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

def get_and_fit_ARDRBF(
    Xs, Ys, Yvars, task_features=None, fidelity_features=None, refit_model=None, state_dict=None,
    fidelity_model_id=None, metric_names=None,
):
    m = ARDRBFGP(train_X=Xs[0], train_Y=Ys[0], train_Yvar=Yvars[0])
    mll = ExactMarginalLogLikelihood(m.likelihood, m)
    mll = fit_gpytorch_model(mll)
    return m

def fit_and_predict_ARDRBF(B, train_X, train_Y, train_Yvar, test_X, mu, sigma):
    # Project training data down to the embedding
    BX = train_X @ B.t()
    m = BotorchModel(model_constructor=get_and_fit_ARDRBF)
    # Fit ARD RBF model on data in embedding
    m.fit([BX], [train_Y], [train_Yvar], [], [], [], [], [])
    # Predict on test points in embedding
    f, var = m.predict(test_X @ B.t())
    # Return predictions, un-standardized
    return f.squeeze() * sigma + mu, var.squeeze() * sigma ** 2

def run_simulation():
    D = 100
    d = 6
    ntrain = 100
    ntest = 50
    
    # Get projection
    torch.manual_seed(1000)
    B0 = torch.randn(d, D, dtype=torch.double)
    B = B0 / torch.sqrt((B0 ** 2).sum(dim=0))

    # Get fixed train/test data
    train_X, train_Y, train_Yvar, test_X, test_Y, mu, sigma = gen_train_test_sets(B, ntrain, ntest)

    # Predict with each model
    f1, var1 = fit_and_predict_alebo(B, train_X, train_Y, train_Yvar, test_X, mu, sigma)
    f2, var2 = fit_and_predict_map(B, train_X, train_Y, train_Yvar, test_X, mu, sigma)
    f3, var3 = fit_and_predict_ARDRBF(B, train_X, train_Y, train_Yvar, test_X, mu, sigma)

    # Save outcome
    with open('data/fig3_sim_output.pckl', 'wb') as fout:
        pickle.dump((test_Y, f1, var1, f2, var2, f3, var3), fout)

def make_fig_3():
    # Load in simulation results
    with open('data/fig3_sim_output.pckl', 'rb') as fin:
        (test_Y, f1, var1, f2, var2, f3, var3) = pickle.load(fin)
    
    fig = plt.figure(figsize=(3.25, 1.5))

    ax = fig.add_subplot(121)
    ax.errorbar(
        x=test_Y.numpy(), y=f3.numpy(), yerr = 2 * np.sqrt(var3.numpy()),
        c='gray', lw=1, ls='', marker='.', mfc='k', mec='k', ms=3
    )
    x0 = -2.5
    x1 = 0.5
    ax.plot([x0, x1], [x0, x1],  '-', zorder=-5, alpha=0.5, c='steelblue', lw=2)
    ax.set_xlim([x0, x1])
    ax.set_ylim([x0, x1])
    ax.set_yticks([0, -1, -2])
    ax.set_xlabel('True value', fontsize=9)
    ax.set_ylabel('Model prediction', fontsize=9)
    ax.set_title('ARD RBF', fontsize=9)

    ax = fig.add_subplot(122)
    ax.errorbar(
        x=test_Y.numpy(), y=f1.numpy(), yerr = 2 * np.sqrt(var1.numpy()),
        c='gray', lw=1, ls='', marker='.', mfc='k', mec='k', ms=3
    )
    x0 = -3
    x1 = 1
    ax.plot([x0, x1], [x0, x1],  '-', zorder=-5, alpha=0.5, c='steelblue', lw=2)
    ax.set_xlim([x0, x1])
    ax.set_ylim([x0, x1])
    ax.set_title('Mahalanobis', fontsize=9)
    ax.set_xticks([-3, -2, -1, 0, 1])
    ax.set_xlabel('True value', fontsize=9)

    plt.subplots_adjust(right=0.99, bottom=0.23, left=0.13, top=0.88, wspace=0.3)
    plt.savefig('pdfs/ard_mahalanobis.pdf', pad_inches=0)

if __name__ == '__main__':
    #run_simulation()  # This will take ~20s, produces data/fig3_sim_output.pckl
    make_fig_3()
