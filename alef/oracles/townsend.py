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

import numpy as np
from alef.oracles.base_oracle import Standard2DOracle

"""
https://en.wikipedia.org/wiki/Test_functions_for_optimization
Townsend, Alex (January 2014).
"Constrained optimization in Chebfun". chebfun.org.
Retrieved 2017-08-29.
"""


class TownsendMain(Standard2DOracle):
    def __init__(self, observation_noise: float):
        super().__init__(observation_noise, 0.0, 1.0)

    def x_scale(self, x):
        r"""
        rescale x as if we are considering input in [-2.25, 2.25] x [-2.5, 1.75]
        """
        assert x.ndim == 1
        a, b = self.get_box_bounds()
        return (x - a) / (b - a) * np.array([4.5, 4.25]) - np.array([2.25, 2.5])

    def f(self, x1, x2):
        x1, x2 = self.x_scale(np.array([x1, x2]))
        return -((np.cos((x1 - 0.1) * x2)) ** 2) - x1 * np.sin(3 * x1 + x2)

    def query(self, x, noisy=True):
        function_value = self.f(x[0], x[1])
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value


class TownsendConstraint(Standard2DOracle):
    def __init__(self, observation_noise: float):
        super().__init__(observation_noise, 0.0, 1.0)

    def x_scale(self, x):
        r"""
        rescale x as if we are considering input in [-2.25, 2.25] x [-2.5, 1.75]
        """
        assert x.ndim == 1
        a, b = self.get_box_bounds()
        return (x - a) / (b - a) * np.array([4.5, 4.25]) - np.array([2.25, 2.5])

    def c(self, x1, x2):
        x1, x2 = self.x_scale(np.array([x1, x2]))
        t = np.arctan2(x1, x2)
        return (
            (2 * np.cos(t) - 0.5 * np.cos(2 * t) - 0.25 * np.cos(3 * t) - 0.125 * np.cos(4 * t)) ** 2
            + (2 * np.sin(t)) ** 2
            - x1**2
            - x2**2
        )

    def query(self, x, noisy=True):
        function_value = self.c(x[0], x[1])
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value


if __name__ == "__main__":
    oracle = TownsendMain(0.01)
    X, Y = oracle.get_random_data_in_box(100, [0.1, 0.2], 0.5, noisy=True)

    print(X.shape)
    print(X.min(axis=0))
    print(X.max(axis=0))
    print(Y.shape)

    oracle.plot()
    oracle = TownsendConstraint(0.01)
    oracle.plot()
