# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

"""
Run benchmarks for LineBO on the Hartmann6 100 problem.

These benchmarks are run using the benchmarking suite contained in the LineBO
codebase. The code here loads in those results and extracts just the function
evaluations, which are then stored in json. There are several steps that must
be done before this script can be run.

1) Clone and install the LineBO software from https://github.com/jkirschner42/LineBO
2) copy data/hartmann6_100.yaml from here into the LineBO/config/ directory.
This problem configuration is based on LineBO/config/hartmann6_sub14.yaml.
3) Run the experiments by executing:

febo create hartmann6_100 --config config/hartmann6_100.yaml
febo run hartmann6_100

4) Copy LineBO/runs/hartmann6_100/data/evaluations.hdf5 into results/
"""

import numpy as np
import h5py
import json

f = h5py.File('results/evaluations.hdf5', 'r')
methods = ['RandomLineBO', 'CoordinateLineBO', 'DescentLineBO', 'Random', 'CMA-ES']

ys = {}
for i, m in enumerate(methods):
    ys[m] = np.zeros((50, 200))
    for rep in range(50):
        ys[m][rep, :] = f[str(i)][str(rep)]['y_exact'] * -3.32237

with open('results/hartmann6_100_randomlinebo.json', 'w') as fout:
    json.dump(ys['RandomLineBO'].tolist(), fout)

with open('results/hartmann6_100_coordinatelinebo.json', 'w') as fout:
    json.dump(ys['CoordinateLineBO'].tolist(), fout)

with open('results/hartmann6_100_descentlinebo.json', 'w') as fout:
    json.dump(ys['DescentLineBO'].tolist(), fout)
