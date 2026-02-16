# Copyright (c) 2024 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import math
import numpy as np
from alef.oracles.base_oracle import Standard2DOracle

"""
Gramacy, R. B., Gray, G. A., Le Digabel, S., Lee, H. K., Ranjan, P., Wells, G., and Wild, S. M. (2016).
Modeling an augmented lagrangian for blackbox constrained optimization.
Technometrics, 58(1):1–11.
"""


class LSQMain(Standard2DOracle):
    def __init__(self, observation_noise: float):
        super().__init__(observation_noise, 0.0, 1.0)

    def f(self, x1, x2):
        return x1 + x2

    def query(self, x, noisy=True):
        function_value = self.f(x[0], x[1])
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value


class LSQConstraint1(Standard2DOracle):
    def __init__(self, observation_noise: float):
        super().__init__(observation_noise, 0.0, 1.0)

    def c(self, x1, x2):
        return x1 + 2 * x2 + 0.5 * np.sin(2 * math.pi * (x1**2 - 2 * x2) - 1.5)

    def query(self, x, noisy=True):
        function_value = self.c(x[0], x[1])
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value


class LSQConstraint2(Standard2DOracle):
    def __init__(self, observation_noise: float):
        super().__init__(observation_noise, 0.0, 1.0)

    def c(self, x1, x2):
        return 1.5 - x1**2 - x2**2

    def query(self, x, noisy=True):
        function_value = self.c(x[0], x[1])
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value


if __name__ == "__main__":
    oracle = LSQMain(0.01)
    oracle.plot()
    oracle = LSQConstraint1(0.01)
    oracle.plot()
    oracle = LSQConstraint2(0.01)
    oracle.plot()
