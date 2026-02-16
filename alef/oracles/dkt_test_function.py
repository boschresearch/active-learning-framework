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

from enum import Enum
import numpy as np
from alef.oracles.base_oracle import Standard1DOracle


class DKTOracleType(Enum):
    STANDARD = 0
    NONSTATIONARY = 1


class DKTTestOracle(Standard1DOracle):
    def __init__(self, task_index: int, oracle_type: DKTOracleType = DKTOracleType.STANDARD) -> None:
        self.oracle_type = oracle_type
        a = 0
        if self.oracle_type == DKTOracleType.STANDARD:
            b = 1
        elif self.oracle_type == DKTOracleType.NONSTATIONARY:
            b = 1.41
        super().__init__(0.01, a, b)
        self.task_index = task_index

    def f(self, x):
        if self.oracle_type == DKTOracleType.STANDARD:
            return (float(self.task_index) / 5.0) * np.sin(4.0 * x) + 0.3 * np.sin(10.0 * x)
        elif self.oracle_type == DKTOracleType.NONSTATIONARY:
            return (float(self.task_index) / 5.0) * np.sin(4.0 * np.power(1.1 * x, 4.0)) + 0.3 * np.sin(10.0 * x)

    def query(self, x, noisy=True):
        function_value = self.f(x)
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value


if __name__ == "__main__":
    from alef.utils.plotter import Plotter

    plotter_object = Plotter(1)
    for i in range(4, 15):
        test_oracle = DKTTestOracle(i, oracle_type=DKTOracleType.NONSTATIONARY)
        X, y = test_oracle.get_random_data(400, False)
        print(y.shape)
        X_noisy, y_noisy = test_oracle.get_random_data(100, True)

        plotter_object.add_gt_function(np.squeeze(X), np.squeeze(y), "blue", 0)
    # plotter_object.add_datapoints(X_noisy, y_noisy, "green", 0)
    plotter_object.show()

    X, y = test_oracle.get_random_data_in_box(400, -1, 2, True)
    print(y.shape)
    print(X.min())
    print(X.max())
