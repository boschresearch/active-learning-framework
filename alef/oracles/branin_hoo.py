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


class BraninHoo(Standard2DOracle):
    def __init__(
        self,
        observation_noise: float = 0.1,
        constants: np.ndarray = np.array([1, 5.1 / 4 / (math.pi**2), 5 / math.pi, 6, 10, 1 / 8 / math.pi]),
    ):
        self._constants = np.squeeze(constants)
        assert self._constants.shape == (6,)
        super().__init__(observation_noise, 0.0, 1.0)

    def sample_constants(self):
        # values designed according to
        # Tighineanu et al. AISTATS 2022 Transfer Learning with Gaussian Processes for Bayesian Optimization
        # Rothfuss et al. CoRL 2022 Meta-Learning Priors for Safe Bayesian Optimization
        # Note: these 2 papers use same bounding values
        a = np.random.uniform(low=0.5, high=1.5)
        b = np.random.uniform(low=0.1, high=0.15)
        c = np.random.uniform(low=1.0, high=2.0)
        r = np.random.uniform(low=5.0, high=7.0)
        s = np.random.uniform(low=8.0, high=12.0)
        t = np.random.uniform(low=0.03, high=0.05)
        return np.array([a, b, c, r, s, t])

    def x_scale(self, x: np.ndarray):
        r"""
        rescale x=[x1, x2] as if we are considering x1 in [-5, 10] & x2 in [0, 15]
        """
        assert x.ndim == 1
        assert len(x) == 2
        a, b = self.get_box_bounds()
        return (x - a) / (b - a) * 15.0 - np.array([5.0, 0.0])

    def f(self, x1, x2):
        a, b, c, r, s, t = self._constants
        x1, x2 = self.x_scale(np.array([x1, x2]))
        f = a * (x2 - b * (x1**2) + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        return f

    def query(self, x, noisy=True):
        function_value = self.f(x[0], x[1])
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value

    def get_function_expression(self):
        a, b, c, r, s, t = self._constants
        print(f"{a}(x2 - {b} x1^2 + {c} x1 - {r})^2 + {s}(1-{t}) np.cos(x1) + {s}")


if __name__ == "__main__":
    oracle = BraninHoo(0.1, normalize_output=True)
    oracle.plot_color_map(0)
    for _ in range(10):
        o_resample = BraninHoo(0.1, constants=oracle.sample_constants(), normalize_output=True)
        o_resample.plot_color_map(0)
