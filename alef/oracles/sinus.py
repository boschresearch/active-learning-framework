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
from alef.oracles.base_oracle import Standard1DOracle


class Sinus(Standard1DOracle):
    def __init__(self, observation_noise=0.01):
        super().__init__(observation_noise, 0.0, 1.0)

    def f(self, x):
        return np.sin(20 * x)

    def query(self, x, noisy=True):
        function_value = self.f(x)
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value

    def get_random_data_out_of_range(self, n: int, noisy: bool = True):
        a, b = self.get_box_bounds()
        X = np.random.uniform(low=b, high=b + 0.5, size=(n, self.get_dimension()))
        function_values = [np.reshape(self.query(x, noisy), [1, -1]) for x in X]
        return X, np.concatenate(function_values, axis=0)


if __name__ == "__main__":
    from alef.utils.plotter import Plotter

    safe_test_func = Sinus(
        0.01,
    )
    print(safe_test_func.f(-3.5))
    print(safe_test_func.query(-3.5))
    # X, y = safe_test_func.get_random_data(1000, False)
    X, y = safe_test_func.get_random_data_in_box(1000, 0.3, 0.5, False)
    # X, y = safe_test_func.get_random_data_out_of_range(1000, False)
    plotter_object = Plotter(1)
    plotter_object.add_gt_function(np.squeeze(X), np.squeeze(y), "blue", 0)
    # plotter_object.add_datapoints(x_initial, y_initial, "green", 0)
    # plotter_object.add_hline(-1.0, "red", 0)
    plotter_object.show()
