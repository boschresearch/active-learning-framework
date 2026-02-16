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


class Griewangk(StandardOracle):
    def __init__(self, observation_noise: float, dimension: int):
        super().__init__(observation_noise, -2.0, 2.0, dimension)

    def x_scale(self, x):
        a, b = self.get_box_bounds()
        return (x - a) / (b - a) * 10.0

    def f(self, x):
        sum_term = 0.0
        product_term = 1.0
        dim = self.get_dimension()
        for i in range(dim):
            xi = self.x_scale(x[i])
            sum_term += (xi**2) / 4000
            product_term *= np.cos(xi / np.sqrt(i + 1))
        return sum_term - product_term + 1

    def query(self, x, noisy=True):
        function_value = self.f(x)
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    oracle = Griewangk(0.01, 2)
    X, Y = oracle.get_random_data_in_box(100, [-0.8, -0.2], 0.5, noisy=True)

    print(X.shape)
    print(X.min(axis=0))
    print(X.max(axis=0))
    print(Y.shape)

    xs, ys = oracle.get_random_data(2000, True)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(xs[:, 0], xs[:, 1], ys, marker=".", color="black")
    plt.show()
