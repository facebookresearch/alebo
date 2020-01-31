# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import json
import numpy as np

from ax.storage.json_store.decoder import object_from_json


def table_S1_data():
    with open('../benchmarks/results/all_aggregated_results.json', 'r') as fin:
        res = object_from_json(json.load(fin))

    for D in [100, 1000]:
        pname = f'Hartmann6, D={D}'
        print('-----', pname)
        for m, ts in res[pname].gen_times.items():
            # Get average total time for fit and gen
            t = np.mean(ts)
            t += np.mean(res[pname].fit_times[m])
            # Divide by 200 to be time per iteration
            t /= 200.
            print(f'{m}: {t}')


if __name__ == '__main__':
    table_S1_data()
