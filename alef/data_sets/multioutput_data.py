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


class MultiOutputDataset:
    def __init__(self):
        self.length = 3000

    def load_data_set(self):
        X = np.random.rand(self.length)[:, None] * 10 - 5
        G = np.hstack((0.5 * np.sin(3 * X) + X, 3.0 * np.cos(X) - X))
        W = np.array([[0.5, -0.3, 1.5], [-0.4, 0.43, 0.0]])
        F = np.matmul(G, W)
        Y = F + np.random.randn(*F.shape) * [0.2, 0.2, 0.2]
        self.x = X
        self.y = Y
        print(self.x.shape)
        print(self.y.shape)

    def get_complete_dataset(self):
        return self.x, self.y

    def sample(self, n, random_x=None, expand_dims=None):
        indexes = np.random.choice(self.length, n, replace=False)
        x_sample = self.x[indexes]
        y_sample = self.y[indexes]
        return x_sample, y_sample


if __name__ == "__main__":
    data = MultiOutputDataset()
    data.load_dataset()
    x, y = data.sample(100)
    print(x.shape)
