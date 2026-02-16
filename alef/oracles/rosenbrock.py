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
from alef.oracles.base_oracle import StandardOracle


class Rosenbrock(StandardOracle):
    """
    see https://www.sfu.ca/~ssurjano/rosen.html
    """

    def __init__(
        self,
        observation_noise: float,
        dimension: int = 4,
    ):
        super().__init__(observation_noise, 0.0, 1.0, dimension)

    def x_scale(self, x: np.ndarray):
        r"""
        rescale x=[x1,..., xd] as if we are considering x in [-5, 10]^d
        """
        assert x.ndim == 1
        assert len(x) == self.get_dimension()
        a, b = self.get_box_bounds()
        return (x - a) / (b - a) * 15.0 - 5

    def f(self, x):
        D = self.get_dimension()
        assert x.ndim == 1
        assert len(x) == D
        x_ = self.x_scale(x)
        out = 0

        for i in range(D - 1):
            out += 100 * (x_[i + 1] - x_[i] ** 2) ** 2 + (x_[i] - 1) ** 2
        return out

    def return_minimum(self):
        D = self.get_dimension()
        x_min_scaled = np.ones(D, type=float)
        a, b = self.get_box_bounds()
        x_min = (x_min_scaled + 5) / 15 * (b - a) + a
        return (x_min, 0)

    def query(self, x, noisy=True):
        function_value = self.f(x)
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value


class Rosenbrock3D(Rosenbrock):
    """
    see https://www.sfu.ca/~ssurjano/rosen.html
    """

    def __init__(
        self,
        observation_noise: float,
    ):
        super().__init__(observation_noise, 3)


class Rosenbrock4D(Rosenbrock):
    """
    see https://www.sfu.ca/~ssurjano/rosen.html
    """

    def __init__(
        self,
        observation_noise: float,
    ):
        super().__init__(observation_noise, 4)


if __name__ == "__main__":
    oracle = Rosenbrock4D(0.01)
    X, Y = oracle.get_random_data_in_box(100, 0, 0.5, noisy=True)
    print(X.shape)
    print(Y.shape)
