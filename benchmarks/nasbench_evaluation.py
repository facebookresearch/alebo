# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

"""
Requires nasbench==1.0 from https://github.com/google-research/nasbench
Also requires dataset nasbench_only108.tfrecord to be downloaded here.

Creates an evaluation functionn for neural architecture search
"""
import numpy as np

from ax.service.ax_client import AxClient

from nasbench.lib.model_spec import ModelSpec
from nasbench import api

nasbench = api.NASBench('nasbench_only108.tfrecord')


def get_spec(adj_indxs, op_indxs):
    """
    Construct a NASBench spec from adjacency matrix and op indicators
    """
    op_names = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    ops = ['input']
    ops.extend([op_names[i] for i in op_indxs])
    ops.append('output')
    iu = np.triu_indices(7, k=1)
    adj_matrix = np.zeros((7, 7), dtype=np.int32)
    adj_matrix[(iu[0][adj_indxs], iu[1][adj_indxs])] = 1
    spec = ModelSpec(adj_matrix, ops)
    return spec


def evaluate_x(x):
    """
    Evaluate NASBench on the model defined by x.

    x is a 36-d array.
    The first 21 are for the adjacency matrix. Largest entries will have the
    corresponding element in the adjacency matrix set to 1, with as many 1s as
    possible within the NASBench model space.
    The last 15 are for the ops in each of the five NASBench model components.
    One-hot encoded for each of the 5 components, 3 options.
    """
    assert len(x) == 36
    x_adj = x[:21]
    x_op = x[-15:]
    x_ord = x_adj.argsort()[::-1]
    op_indxs = x_op.reshape(3, 5).argmax(axis=0).tolist()
    last_good = None
    for i in range(1, 22):
        model_spec = get_spec(x_ord[:i], op_indxs)
        if model_spec.matrix is not None:
            # We have a connected graph
            # See if it has too many edges
            if model_spec.matrix.sum() > 9:
                break
            last_good = model_spec
    if last_good is None:
        # Could not get a valid spec from this x. Return bad metric values.
        return [0.80], [50 * 60]
    fixed_metrics, computed_metrics = nasbench.get_metrics_from_spec(last_good)
    test_acc = [r['final_test_accuracy'] for r in computed_metrics[108]]
    train_time = [r['final_training_time'] for r in computed_metrics[108]]
    return np.mean(test_acc), np.mean(train_time)


def evaluate_parameters(parameters):
    x = np.array([parameters[f'x{i}'] for i in range(36)])
    test_acc, train_time = evaluate_x(x)
    return {
        'final_test_accuracy': (test_acc, 0.0),
        'final_training_time': (train_time, 0.0),
    }


def get_nasbench_ax_client(generation_strategy):
    # Get parameters
    parameters = [
        {
            "name": f"x{i}",
            "type": "range",
            "bounds": [0, 1],
            "value_type": "float",
            "log_scale": False,
        } for i in range(36)
    ]
    axc = AxClient(generation_strategy=generation_strategy, verbose_logging=False)
    axc.create_experiment(
        name="nasbench",
        parameters=parameters,
        objective_name="final_test_accuracy",
        minimize=False,
        outcome_constraints=["final_training_time <= 1800"],
    )
    return axc


class NASBenchRunner:
    """
    A runner for non-Ax methods.
    Assumes method MINIMIZES.
    """
    def __init__(self, max_eval):
        # For tracking iterations
        self.fs = []
        self.feas = []
        self.n_eval = 0
        self.max_eval = max_eval

    def f(self, x):
        if self.n_eval >= self.max_eval:
            raise ValueError("Evaluation budget exhuasted")
        test_acc, train_time = evaluate_x(x)
        feas = bool(train_time <= 1800)
        if not feas:
            val = 0.80  # bad value for infeasible
        else:
            val = test_acc
        self.n_eval += 1
        self.fs.append(test_acc)  # Store the true, not-negated value
        self.feas.append(feas)
        return -val  # ASSUMES METHOD MINIMIZES
