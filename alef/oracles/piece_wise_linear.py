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


from alef.oracles.base_oracle import BaseOracle
import numpy as np
from typing import Tuple
from alef.utils.plotter import Plotter
from enum import Enum


class FunctionType(Enum):
    FUNC1 = 1
    FUNC2 = 2
    FUNC3 = 3


class PieceWiseLinear(BaseOracle):
    def __init__(self, observation_noise: float, function_type: FunctionType) -> None:
        self.observation_noise = observation_noise
        self.__a = -2
        self.__b = 2
        self.__dimension = 1
        self.__f_a = 0
        self.scale = 3
        if function_type == FunctionType.FUNC1:
            self.__c_s = [
                -1.8,
                -1.5,
                -1.3,
                -1.2,
                -1.0,
                -0.8,
                -0.7,
                -0.6,
                -0.5,
                -0.4,
                -0.1,
                0.0,
                0.1,
                0.5,
                0.7,
                0.8,
                0.9,
                1.0,
                1.2,
                1.4,
                1.6,
                1.8,
            ]
            self.__a_s = [
                0.1,
                0.2,
                0.3,
                0.2,
                0.1,
                0.0,
                -0.05,
                -0.2,
                -0.4,
                0.2,
                0.3,
                -0.3,
                0.4,
                -0.3,
                0.2,
                0.1,
                0.0,
                -0.2,
                -0.3,
                -0.4,
                -0.2,
                -0.1,
                0.0,
                0.0,
            ]
        elif function_type == FunctionType.FUNC2:
            self.__c_s = [-1.8, -1.0, 0.0, 1.0, 1.5]
            self.__a_s = [0.1, -0.1, -0.2, 0.2, -0.1, 0.2]
        elif function_type == FunctionType.FUNC3:
            self.__c_s = [-1.8, -1.0, 0.0, 1.0, 1.5]
            self.__a_s = [0.1, -0.1, -0.2, 0.2, -0.1, 0.2]

    def f(self, x):
        const = self.__f_a
        previous_flip_point = self.__a
        for i, c in enumerate(self.__c_s):
            if x <= c:
                f_x = const + self.__a_s[i] * (x - previous_flip_point)
                return f_x
            const += self.__a_s[i] * (c - previous_flip_point)
            previous_flip_point = c
        return const + self.__a_s[-1] * (x - self.__c_s[-1])

    def query(self, x, noisy=True):
        function_value = self.scale * self.f(x)
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value

    def get_box_bounds(self):
        return self.__a, self.__b

    def get_dimension(self):
        return self.__dimension

    def get_random_data(self, n: int, noisy: bool) -> Tuple[np.array, np.array]:
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

        function_values = [np.reshape(self.query(x, noisy), [1, -1]) for x in X]
        return X, np.concatenate(function_values, axis=0)

    def plot(self):
        X, y = self.get_random_data(1000, False)
        plotter_object = Plotter(1)
        plotter_object.add_gt_function(np.squeeze(X), np.squeeze(y), "blue", 0)
        plotter_object.show()


if __name__ == "__main__":
    oracle = PieceWiseLinear(0.01, FunctionType.FUNC1)
    X, Y = oracle.get_random_data_in_box(100, 1, 2, noisy=True)

    print(X.shape)
    print(X.min(axis=0))
    print(X.max(axis=0))
    print(Y.shape)

    oracle.plot()
