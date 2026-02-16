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


class Eggholder(Standard2DOracle):
    def __init__(self, observation_noise: float, constants: np.ndarray = np.array([1.0, 1.0, 47.0])):
        self._constants = np.squeeze(constants)
        assert self._constants.shape == (3,)
        super().__init__(observation_noise, 0.0, 1.0)

    def sample_constants(self):
        # values designed according to
        # Tighineanu et al. AISTATS 2022 Transfer Learning with Gaussian Processes for Bayesian Optimization
        # Rothfuss et al. CoRL 2022 Meta-Learning Priors for Safe Bayesian Optimization
        # Note: these 2 papers use same bounding values
        a = np.random.uniform(low=0.5, high=1.5)
        b = np.random.uniform(low=0.1, high=0.15)
        c = np.random.uniform(low=1.0, high=2.0)
        return np.array([a, b, c])

    def x_scale(self, x):
        r"""
        rescale x as if we are considering input in [-512, 512]
        """
        a, b = self.get_box_bounds()
        return (x - a) / (b - a) * 1024 - 512

    def f(self, x1, x2):
        a, b, c = self._constants
        x1 = self.x_scale(x1)
        x2 = self.x_scale(x2)
        f_raw = -(x2 + c) * np.sin(np.sqrt(abs(a * x2 + x1 / 2 + 47))) - b * x1 * np.sin(np.sqrt(abs(x1 - x2 - 47)))
        return f_raw

    def query(self, x, noisy=True):
        function_value = self.f(x[0], x[1])
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value

    def get_function_expression(self):
        a, b, c = self._constants
        print(f"-(x2 + {c}) sin(sqrt(|{a}x2 + x1/2 + 47|)) - {b}x1 sin(sqrt(|x1-x2-47|))")


if __name__ == "__main__":
    oracle = Eggholder(0.01)
    X, Y = oracle.get_random_data_in_box(100, [0.1, 0.2], 0.5, noisy=True)
    oracle.get_function_expression()
    print(X.shape)
    print(X.min(axis=0))
    print(X.max(axis=0))
    print(Y.shape)

    oracle.plot()
