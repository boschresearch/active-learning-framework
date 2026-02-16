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
from alef.utils.utils import filter_safety
from alef.oracles.base_oracle import Standard2DOracle


class Exponential2D(Standard2DOracle):
    def __init__(self, observation_noise=0.01):
        super().__init__(observation_noise, -2.0, 5.0)

    def f(self, x1, x2):
        return x1 * np.exp(-1 * np.power(x1, 2.0) - np.power(x2, 2.0)) * 0.5

    def query(self, x, noisy=True, scale_factor=1.0):
        function_value = self.f(x[0], x[1]) * scale_factor
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value

    def get_scaled_random_data(self, n, noisy=True, f_scale_factor=10):
        a, b = self.get_box_bounds()
        dim = self.get_dimension()
        X = np.random.uniform(low=a, high=b, size=(n, dim))
        function_values = []
        for x in X:
            function_value = self.query(x, noisy, f_scale_factor)
            function_values.append(function_value)
        X = (X - a) / (b - a)
        return X, np.expand_dims(np.array(function_values), axis=1)

    def get_random_data_in_random_box_with_safety(self, n, box_width, safety_threshold, safety_is_upper_bound=False):
        a, b = self.get_box_bounds()
        dim = self.get_dimension()
        n_safe = 0
        while n_safe < n:
            a_loop = np.random.uniform(low=a, high=b - box_width, size=dim)
            X, y = self.get_random_data_in_box(3 * n, a_loop, box_width)
            X_safe, y_safe = filter_safety(X, y, safety_threshold, safety_is_upper_bound)
            n_safe = len(X_safe)
        chosen_indexes = np.random.choice(np.arange(0, n_safe), n, replace=False)
        # print(X_safe[chosen_indexes])
        return X_safe[chosen_indexes], y_safe[chosen_indexes]


if __name__ == "__main__":
    function = Exponential2D(0.01)
    X, y = function.get_random_data_in_random_box_with_safety(10, 1.0, -0.1)
    print(X)
    print(y)
    print(function.query(np.array([0.0, 0.0])))
    X, y = function.get_random_data(100)
    print(X.shape)
    print(y.shape)
    function.plot()
