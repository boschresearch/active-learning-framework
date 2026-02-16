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
from alef.oracles.base_oracle import Standard1DOracle, Standard2DOracle, StandardOracle


class Flexible1DOracle(Standard1DOracle):
    def __init__(self, observation_noise=0.01):
        super().__init__(observation_noise, 0.0, 1.0)

    def set_f(self, function):
        self.f = function

    def query(self, x, noisy=True):
        function_value = self.f(x)
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value


class Flexible2DOracle(Standard2DOracle):
    def __init__(self, observation_noise=0.01):
        super().__init__(observation_noise, 0.0, 1.0)

    def set_f(self, function):
        self.f = function

    def query(self, x, noisy=True):
        function_value = self.f(x)
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value


class FlexibleOracle(StandardOracle):
    def __init__(self, observation_noise=0.01):
        super().__init__(observation_noise, 0.0, 1.0)

    def set_f(self, function):
        self.f = function

    def query(self, x, noisy=True):
        function_value = self.f(x)
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value
