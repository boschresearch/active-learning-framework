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
from alef.data_sets.base_data_set import StandardDataSet


class PytestSet(StandardDataSet):
    def __init__(self, base_path: str = ""):
        self.file_path = ""
        self.name = "numpy_arrays"

    def load_data_set(self):
        self.x = np.arange(3000).reshape([1000, 3]) / 3000
        self.y = np.sin(10 * self.x[:, 0, None]) + np.cos(10 * self.x[:, 1, None]) + np.sin(10 * self.x[:, 2, None])
        self.length = 1000


class PytestMOSet(StandardDataSet):
    def __init__(self, base_path: str = ""):
        self.file_path = ""
        self.name = "numpy_arrays"

    def load_data_set(self):
        self.x = np.hstack(
            [
                np.linspace(-2, 1, 1000, endpoint=True).reshape([1000, 1]),
                np.linspace(-1, 1, 1000, endpoint=True).reshape([1000, 1]),
                np.linspace(-1, 2, 1000, endpoint=True).reshape([1000, 1]),
            ]
        )
        self.y = np.empty([1000, 2])
        self.y[:, 0] = np.sin(10 * self.x[:, 0]) + np.cos(10 * self.x[:, 1]) + np.sin(10 * self.x[:, 2])
        self.y[:, 1] = np.cos(10 * self.x[:, 0]) + np.sin(10 * self.x[:, 1]) + np.cos(10 * self.x[:, 2])
        self.length = 1000


if __name__ == "__main__":
    dataset = PytestSet()
    dataset.load_data_set()
