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
import logging
from alef.oracles.base_oracle import StandardOracle, Standard1DOracle, Standard2DOracle


class OracleNormalizer(StandardOracle):
    def __init__(self, oracle: StandardOracle):
        self._oracle = oracle
        a, b = oracle.get_box_bounds()
        observation_noise = oracle.observation_noise
        dimension = oracle.get_dimension()
        super().__init__(observation_noise, a, b, dimension)
        self._normalize_mean = 1.0
        self._normalize_scale = 1.0

    def get_normalization(self):
        return self._normalize_mean, self._normalize_scale

    def set_normalization_manually(self, normalize_mean: float, normalize_scale: float):
        assert normalize_scale > 0
        self._normalize_mean = normalize_mean
        self._normalize_scale = normalize_scale

    def set_normalization_by_sampling(self):
        X, Y = self._oracle.get_random_data(5000, noisy=False)
        self._normalize_mean = Y.mean()
        self._normalize_scale = math.sqrt(Y.var())

    def query(self, x, noisy=True):
        """main method for querying the oracle"""
        f = self._oracle.query(x, noisy=False)
        function_value = (f - self._normalize_mean) / self._normalize_scale
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value

    def plot(self, *args, **kwargs):
        if isinstance(self._oracle, (Standard1DOracle, Standard2DOracle)):
            logging.warning("The plot scale is not normalized!")
            return self._oracle.plot(*args, **kwargs)
        else:
            raise NotImplementedError

    def save_plot(self, *args, **kwargs):
        if isinstance(self._oracle, (Standard1DOracle, Standard2DOracle)):
            logging.warning("The plot scale is not normalized!")
            return self._oracle.save_plot(*args, **kwargs)
        else:
            raise NotImplementedError

    def plot_color_map(self, *args, **kwargs):
        if isinstance(self._oracle, Standard2DOracle):
            logging.warning("The plot scale is not normalized!")
            return self._oracle.plot_color_map(*args, **kwargs)
        else:
            raise NotImplementedError

    def save_color_map(self, *args, **kwargs):
        if isinstance(self._oracle, Standard2DOracle):
            logging.warning("The plot scale is not normalized!")
            return self._oracle.save_color_map(*args, **kwargs)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    from alef.oracles import BraninHoo

    oracle = OracleNormalizer(BraninHoo(0.1))
    oracle.set_normalization_by_sampling()
    mu, std = oracle.get_normalization()
    print(mu, std)
    oracle.plot()
