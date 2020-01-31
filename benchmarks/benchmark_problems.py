# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import json
import numpy as np
from scipy.stats import special_ortho_group
from typing import List, Optional

from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.metrics.branin import BraninMetric
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.storage.metric_registry import register_metric
from ax.utils.measurement.synthetic_functions import hartmann6


### Hartmann6 problem, D=100 and D=1000

# Relevant parameters were chosen randomly using
# x = np.arange(100)
# np.random.seed(10)
# np.random.shuffle(x)
# print(x[:6])  # [19 14 43 37 66  3]

hartmann6_100 = BenchmarkProblem(
    name="Hartmann6, D=100",
    optimal_value=-3.32237,
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=Hartmann6Metric(
                name="objective",
                param_names=["x19", "x14", "x43", "x37", "x66", "x3"],
                noise_sd=0.0,
            ),
            minimize=True,
        )
    ),
    search_space=SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i}", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
            )
            for i in range(100)
        ]
    ),
)

            
hartmann6_1000 = BenchmarkProblem(
    name="Hartmann6, D=1000",
    optimal_value=-3.32237,
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=Hartmann6Metric(
                name="objective",
                param_names=["x190", "x140", "x430", "x370", "x660", "x30"],
                noise_sd=0.0,
            ),
            minimize=True,
        )
    ),
    search_space=SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i}", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
            )
            for i in range(1000)
        ]
    ),
)


### Branin problem, D=100 and sensitivity analysis

# Original x1 and x2 have different bounds, so we create blocks of 50 for each
# with each of the bounds and set the relevant parameters in those blocks.

branin_100 = BenchmarkProblem(
    name="Branin, D=100",
    optimal_value=0.397887,
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=BraninMetric(
                name="objective", param_names=["x19", "x64"], noise_sd=0.0
            ),
            minimize=True,
        )
    ),
    search_space=SearchSpace(
        parameters=[  # pyre-ignore
            RangeParameter(
                name=f"x{i}", parameter_type=ParameterType.FLOAT, lower=-5.0, upper=10.0
            )
            for i in range(50)
        ]
        + [
            RangeParameter(
                name=f"x{i + 50}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=15.0,
            )
            for i in range(50)
        ]
    ),
)
            
# Additional dimensionalities for the sensitivity analysis to D.
# Random embedding methods are invariant to the ordering of relevant/irrelevant
# parameters, and also to the bounds on the irrelevant parameters. So since
# these problems are being used only for ALEBO, we can simplify their
# definition and take x0 and x1 as the relevant.

base_branin_parameters = [
    RangeParameter(
        name="x0", parameter_type=ParameterType.FLOAT, lower=-5.0, upper=10.0
    ),
    RangeParameter(
        name="x1", parameter_type=ParameterType.FLOAT, lower=0.0, upper=15.0
    ),
]

branin_by_D = {
    D: BenchmarkProblem(
        name="Branin, D=" + str(D),
        optimal_value=0.397887,
        optimization_config=OptimizationConfig(
            objective=Objective(
                metric=BraninMetric(
                    name="objective", param_names=["x0", "x1"], noise_sd=0.0
                ),
                minimize=True,
            )
        ),
        search_space=SearchSpace(
            parameters=base_branin_parameters  # pyre-ignore
            + [
                RangeParameter(
                    name=f"x{i}",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=1.0,
                )
                for i in range(2, D)
            ]
        ),
    )
    for D in [50, 200, 500, 1000]
}


### Gramacy problem, D=100

class GramacyObjective(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return x.sum()


class GramacyConstraint1(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return 1.5 - x[0] - 2 * x[1] - 0.5 * np.sin(2 * np.pi * (x[0] ** 2 - 2 * x[1]))


class GramacyConstraint2(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return x[0] ** 2 + x[1] ** 2 - 1.5


# Register these metrics so they can be serialized to json
register_metric(metric_cls=GramacyObjective, val=101)
register_metric(metric_cls=GramacyConstraint1, val=102)
register_metric(metric_cls=GramacyConstraint2, val=103)


gramacy_100 = BenchmarkProblem(
    name="Gramacy, D=100",
    optimal_value=0.5998,
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=GramacyObjective(
                name="objective", param_names=["x19", "x64"], noise_sd=0.0
            ),
            minimize=True,
        ),
        outcome_constraints=[
            OutcomeConstraint(
                metric=GramacyConstraint1(
                    name="constraint1", param_names=["x19", "x64"], noise_sd=0.0
                ),
                op=ComparisonOp.LEQ,
                bound=0.0,
                relative=False,
            ),
            OutcomeConstraint(
                metric=GramacyConstraint2(
                    name="constraint2", param_names=["x19", "x64"], noise_sd=0.0
                ),
                op=ComparisonOp.LEQ,
                bound=0.0,
                relative=False,
            ),
        ],
    ),
    search_space=SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i}", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
            )
            for i in range(100)
        ]
    ),
)

### Hartmann6 D=1000 with random subspace

class Hartmann6RandomSubspace1000Metric(NoisyFunctionMetric):
    def __init__(
        self,
        name: str,
        param_names: List[str],
        noise_sd: float = 0.0,
        lower_is_better: Optional[bool] = None,
    ) -> None:
        super().__init__(
            name=name,
            param_names=param_names,
            noise_sd=noise_sd,
            lower_is_better=lower_is_better,
        )
        # Set the random basis
        try:
            with open('data/random_subspace_1000x6.json', 'r') as fin:
                self.random_basis = np.array(json.load(fin))
        except IOError:
            np.random.seed(1000)
            self.random_basis = special_ortho_group.rvs(1000)[:6, :]
            with open('data/random_subspace_1000x6.json', 'w') as fout:
                json.dump(self.random_basis.tolist(), fout)

    def f(self, x: np.ndarray) -> float:
        # Project down to the true subspace
        z = self.random_basis @ x
        # x \in [-1, 1], so adjust z to be closer to [0, 1], and evaluate
        return hartmann6((z + 1) / 2.)

register_metric(metric_cls=Hartmann6RandomSubspace1000Metric, val=104)

hartmann6_random_subspace_1000 = BenchmarkProblem(
    name="Hartmann6 random subspace, D=1000",
    optimal_value=-3.32237,
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=Hartmann6RandomSubspace1000Metric(
                name="objective",
                param_names=[f"x{i}" for i in range(1000)],
                noise_sd=0.0,
            ),
            minimize=True,
        )
    ),
    search_space=SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i}", parameter_type=ParameterType.FLOAT, lower=-1.0, upper=1.0
            )
            for i in range(1000)
        ]
    ),
)
