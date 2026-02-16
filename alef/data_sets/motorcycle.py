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

import sys

sys.path.append(".")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Motorcycle:
    def __init__(self, file_path):
        self.file_path = file_path
        self.observation_noise = 0.02
        self.add_noise = False

    def load_data_set(self):
        df = pd.read_csv(self.file_path, sep=",")
        self.x = df["times"].to_numpy() / 60
        self.y = df["accel"].to_numpy() / 100
        self.x = np.expand_dims(self.x, axis=1)
        self.y = np.expand_dims(self.y, axis=1)
        self.length = self.x.shape[0]

    def get_complete_dataset(self):
        return self.x, self.y

    def sample(self, n, random_x=None, expand_dims=None):
        if n > self.length:
            n = self.length
        indexes = np.random.choice(self.length, n, replace=False)
        x_sample = self.x[indexes]
        f_sample = self.y[indexes]
        noise = np.random.randn(n, 1) * self.observation_noise
        if self.add_noise:
            y_sample = f_sample + noise
        else:
            y_sample = f_sample
        cat_indexes = np.argwhere(x_sample[:, 0] >= 0.2)
        cats_sample = np.repeat(1.0, x_sample[:, 0].shape[0])
        cats_sample[cat_indexes] = 0.0
        return x_sample, f_sample, y_sample, cats_sample

    def plot(self):
        xs, _, ys, cats = self.sample(200)
        fig, ax = plt.subplots()
        ax.plot(xs, ys, ".")
        plt.show()
