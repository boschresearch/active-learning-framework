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
from alef.utils.plotter import Plotter
from alef.oracles.base_oracle import BaseOracle
from enum import Enum


class FunctionType(Enum):
    NONSTATIONARY = 1
    LOW_LENGHTSCALE = 2
    HIGH_LENGTHSCALE = 3


class SafeTestFunc(BaseOracle):
    def __init__(self, observation_noise=0.01, function_type=FunctionType.NONSTATIONARY):
        self.observation_noise = observation_noise
        self.__dimension = 1
        self.__a = -6
        self.__b = 10
        self._function_type = function_type

    def f(self, x):
        if self._function_type == FunctionType.NONSTATIONARY:
            if x <= 0.6:
                return np.sin(x - 3.0) + 0.3 * np.sin(5 * (x - 3.0)) + 0.1 * x - 0.15
            else:
                return (
                    0.8 * np.sin(0.5 * (x - 0.6)) + np.sin(0.6 - 3.0) + 0.3 * np.sin(5 * (0.6 - 3.0)) + 0.1 * x - 0.15
                )
        elif self._function_type == FunctionType.HIGH_LENGTHSCALE:
            x = x + 0.12
            return np.sin(x - 3.0) + 0.3 * np.sin(5 * (x - 3.0)) + 0.2 * x + 0.25 - 0.02 * np.power(x, 2)

        elif self._function_type == FunctionType.LOW_LENGHTSCALE:
            x = x + 3.3
            return 0.8 * np.sin(0.5 * (x - 0.6)) + np.sin(0.6 - 3.0) + 0.3 * np.sin(5 * (0.6 - 3.0)) + 0.1 * x

    def query(self, x, noisy=True):
        function_value = self.f(x)
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value

    def query_batch(self, X, noisy=True):
        function_values = []
        for x in X:
            function_value = self.query(x, noisy)
            function_values.append(function_value)
        return np.array(function_values)

    def get_random_data(self, n, noisy=True):
        X = np.random.uniform(low=self.__a, high=self.__b, size=(n, self.__dimension))
        function_values = []
        for x in X:
            function_value = self.query(x, noisy)
            function_values.append(function_value)
        return X, np.array(function_values)

    def get_random_data_in_box(self, n: int, a: float, box_width: float, noisy: bool = True):
        """
        a and box_width are floats because self.__dimension = 1
        """
        b = a + box_width
        assert a < self.__b
        assert b > self.__a
        X = np.random.uniform(low=max(a, self.__a), high=min(b, self.__b), size=(n, self.get_dimension()))

        function_values = []
        for x in X:
            function_value = self.query(x, noisy)
            function_values.append(function_value)
        return X, np.expand_dims(np.array(function_values), axis=1)

    def get_box_bounds(self):
        return self.__a, self.__b

    def get_dimension(self):
        return self.__dimension

    def get_left_data(self, n, noisy=True):
        X = np.random.uniform(low=self.__a, high=0.6, size=(n, self.__dimension))
        function_values = []
        for x in X:
            function_value = self.query(x, noisy)
            function_values.append(function_value)
        return X, np.array(function_values)

    def get_random_data_outside_intervall(self, a, b, n, noisy=True):
        n1 = int(n / 2)
        n2 = n - n1
        X1 = np.random.uniform(low=self.__a, high=a, size=(n1, self.__dimension))
        X2 = np.random.uniform(low=b, high=self.__b, size=(n2, self.__dimension))
        X = np.concatenate((X1, X2))
        function_values = []
        for x in X:
            function_value = self.query(x, noisy)
            function_values.append(function_value)
        return X, np.array(function_values)

    def get_right_data(self, n, noisy=True):
        X = np.random.uniform(low=0.6, high=self.__b, size=(n, self.__dimension))
        function_values = []
        for x in X:
            function_value = self.query(x, noisy)
            function_values.append(function_value)
        return X, np.array(function_values)

    def get_data_in_box(self, n, a, b, noisy=True):
        X = np.random.uniform(low=a, high=b, size=(n, self.__dimension))
        function_values = []
        for x in X:
            function_value = self.query(x, noisy)
            function_values.append(function_value)
        return X, np.array(function_values)

    def get_data_in_random_box(self, n, box_width, bound_a, bound_b, noisy=True):
        a = np.random.uniform(low=bound_a, high=bound_b - box_width)
        b = a + box_width
        return self.get_data_in_box(n, a, b, noisy=noisy)


if __name__ == "__main__":
    safe_test_func = SafeTestFunc(0.01, FunctionType.NONSTATIONARY)
    print(safe_test_func.f(-3.5))
    print(safe_test_func.query(-3.5))
    X, y = safe_test_func.get_random_data_in_box(1000, -7, 4, False)
    x_initial, y_initial = safe_test_func.get_random_data_outside_intervall(2, 5, 300)
    plotter_object = Plotter(1)
    plotter_object.add_gt_function(np.squeeze(X), np.squeeze(y), "blue", 0)
    plotter_object.add_datapoints(x_initial, y_initial, "green", 0)
    plotter_object.add_hline(-1.0, "red", 0)
    plotter_object.show()
