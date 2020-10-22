# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rc
import matplotlib
rc('font', family='serif', style='normal', variant='normal', weight='normal', stretch='normal', size=8)
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['xtick.labelsize'] = 7
matplotlib.rcParams['ytick.labelsize'] = 7
matplotlib.rcParams['axes.titlesize'] = 9

plot_method_names = [
    'ALEBO (ours)',
    'REMBO',
    'HeSBO, $d_e$=$d$',
    'HeSBO, $d_e$=$2d$',
    'REMBO-$\phi k_{\Psi}$',
    'REMBO-$\gamma k_{\Psi}$',
    'EBO',
    'Add-GP-UCB',
    'SMAC',
    'CMA-ES',
    'TuRBO',
    'Sobol',
    'CoordinateLineBO',
    'RandomLineBO',
    'DescentLineBO',
]

plot_colors={
    'ALEBO (ours)': plt.cm.tab20(0),
    'REMBO': plt.cm.tab20(1),
    'HeSBO, $d_e$=$d$': plt.cm.tab20(2),
    'HeSBO, $d_e$=$2d$': plt.cm.tab20(3),
    'REMBO-$\phi k_{\Psi}$': plt.cm.tab20(4),
    'REMBO-$\gamma k_{\Psi}$': plt.cm.tab20(5),
    'EBO': plt.cm.tab20(6),
    'Add-GP-UCB': plt.cm.tab20(7),
    'SMAC': plt.cm.tab20(8),
    'CMA-ES': plt.cm.tab20(9),
    'TuRBO': plt.cm.tab20(10),
    'Sobol': plt.cm.tab20(14),
    'CoordinateLineBO': plt.cm.tab20(12),
    'RandomLineBO': plt.cm.tab20(16),
    'DescentLineBO': plt.cm.tab20(18),
}
