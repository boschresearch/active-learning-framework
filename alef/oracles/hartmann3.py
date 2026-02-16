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


class Hartmann3(StandardOracle):
    """
    see https://www.sfu.ca/~ssurjano/hart3.html
    """

    def __init__(
        self,
        observation_noise: float,
    ):
        super().__init__(observation_noise, 0.0, 1.0, 3)

    def f(self, x1, x2, x3):
        x = np.array([[x1, x2, x3]])

        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
        P = 0.0001 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
        return -np.sum(alpha * np.exp(-np.sum(A * (x - P) ** 2, axis=-1)), axis=0)

    def return_minimum(self):
        return (np.array([0.114614, 0.555649, 0.852547]), -3.86276)

    def query(self, x, noisy=True):
        function_value = self.f(x[0], x[1], x[2])
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value


if __name__ == "__main__":
    oracle = Hartmann3(0.01)
    X, Y = oracle.get_random_data_in_box(100, 0, 0.5, noisy=True)
    print(X.shape)
    print(Y.shape)
