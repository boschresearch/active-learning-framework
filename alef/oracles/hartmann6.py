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


class Hartmann6(StandardOracle):
    """
    see https://www.sfu.ca/~ssurjano/hart6.html
    """

    def __init__(
        self,
        observation_noise: float,
    ):
        super().__init__(observation_noise, 0.0, 1.0, 6)

    def f(self, x1, x2, x3, x4, x5, x6):
        x = np.array([[x1, x2, x3, x4, x5, x6]])

        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
        )
        P = 0.0001 * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )
        return -1 / 1.94 * (2.58 + np.sum(alpha * np.exp(-np.sum(A * (x - P) ** 2, axis=-1)), axis=0))

    def return_minimum(self):
        return (np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]), -3.042458763)

    def query(self, x, noisy=True):
        function_value = self.f(x[0], x[1], x[2], x[3], x[4], x[5])
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value


if __name__ == "__main__":
    oracle = Hartmann6(0.01)
    X, Y = oracle.get_random_data_in_box(100, 0, 0.5, noisy=True)
    print(X.shape)
    print(Y.shape)
