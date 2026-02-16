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
from typing import Union, Sequence
from alef.utils.utils import check1Dlist
from alef.oracles.base_oracle import BaseOracle
import matplotlib.pyplot as plt


class HighDimAdditive(BaseOracle):
    def __init__(self, observation_noise, dimension_added):
        self.__a = 0
        self.__b = 2
        self.__dimension = dimension_added + 2
        self.observation_noise = observation_noise

    def f(self, x1, x2):
        return np.sin(x1 * 2) * np.cos(x2 * 2)

    def query(self, x, noisy=True, scale_factor=1.0):
        function_value = self.f(x[0], x[1]) * scale_factor
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value

    def get_random_data(self, n, noisy=True):
        X = np.random.uniform(low=self.__a, high=self.__b, size=(n, self.get_dimension()))
        function_values = []
        for x in X:
            function_value = self.query(x, noisy)
            function_values.append(function_value)
        return X, np.expand_dims(np.array(function_values), axis=1)

    def get_random_data_in_box(
        self, n: int, a: Union[float, Sequence[float]], box_width: Union[float, Sequence[float]], noisy: bool = True
    ):
        aa = np.array(check1Dlist(a, self.get_dimension()))
        bw = np.array(check1Dlist(box_width, self.get_dimension()))
        bb = aa + bw
        assert np.all(aa < self.__b)
        assert np.all(bb > self.__a)
        aa[aa < self.__a] = self.__a
        bb[bb > self.__b] = self.__b
        X = np.random.uniform(low=aa, high=bb, size=(n, self.get_dimension()))

        function_values = [np.reshape(self.query(x, noisy), [1, -1]) for x in X]
        return X, np.concatenate(function_values, axis=0)

    def get_box_bounds(self):
        return self.__a, self.__b

    def get_dimension(self):
        return self.__dimension

    def plot(self):
        xs, ys = self.get_random_data(2000, True)
        # x_safe,y_safe = self.get_random_data_in_random_box_with_safety(10,2.0,-0.1)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(xs[:, 0], xs[:, 1], ys, marker=".", color="black")
        # ax.scatter(x_safe[:,0],x_safe[:,1],y_safe,marker='o',color='green')
        plt.show()


if __name__ == "__main__":
    function = HighDimAdditive(0.01, 2)
    function.plot()
    x_data, y_data = function.get_random_data(100)
    print(x_data.shape)
    print(y_data.shape)

    X, Y = function.get_random_data_in_box(100, -0.5, 1, noisy=True)

    print(X.shape)
    print(X.min(axis=0))
    print(X.max(axis=0))
    print(Y.shape)
