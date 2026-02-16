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
from enum import Enum


class OutputType(Enum):
    LIFT_AND_DRAG = 1


class LGBBCombined:
    def __init__(self, observation_noise, file_path):
        self.file_path = file_path
        self.observation_noise = observation_noise
        self.add_noise = True
        self.exclude_outlier = True
        self.output_type = OutputType.LIFT_AND_DRAG
        self.beta = 0.0

    def load_data_set(self):
        df = pd.read_csv(self.file_path, sep=" ")
        df_beta_0 = df[df["beta"] == self.beta]
        if self.output_type == OutputType.LIFT_AND_DRAG:
            key = "drag"
            low = df_beta_0[key].quantile(0.01)
            high = df_beta_0[key].quantile(0.99)
            df_filtered = df_beta_0[(df_beta_0[key] < high) & (df_beta_0[key] > low)]
            x1 = df_filtered["mach"].to_numpy() / 6
            x2 = (df_filtered["alpha"].to_numpy() + 5) / 35
            y_drag = df_filtered[key].to_numpy()
            y_lift = df_filtered["lift"].to_numpy()
            self.d_output = 2
        self.x = np.stack((x1, x2), axis=1)
        self.y = np.stack((y_lift, y_drag), axis=1)
        self.length = x1.shape[0]

    def get_complete_dataset(self):
        n = self.y.shape[0]
        if self.add_noise:
            noise = np.random.randn(n, self.d_output) * self.observation_noise
            y = self.y + noise
        else:
            y = self.y
        return self.x, y

    def sample(self, n):
        indexes = np.random.choice(self.length, n, replace=False)
        x_sample = self.x[indexes]
        f_sample = self.y[indexes]
        noise = np.random.randn(n, self.d_output) * self.observation_noise
        if self.add_noise:
            y_sample = f_sample + noise
        else:
            y_sample = f_sample
        return x_sample, f_sample, y_sample

    def sample_only_one_regime_and_safe(
        self, n, safety_threshold, safety_index, left=False, x1_threshold=0.2, safety_is_upper_bound=True
    ):
        if left:
            x_filtered = self.x[self.x[:, 0] < x1_threshold]
            y_filtered = self.y[self.x[:, 0] < x1_threshold]
        else:
            x_filtered = self.x[self.x[:, 0] >= x1_threshold]
            y_filtered = self.y[self.x[:, 0] >= x1_threshold]
        # y_filtered = np.squeeze(y_filtered)
        if safety_is_upper_bound:
            x_filtered2 = x_filtered[y_filtered[:, safety_index] < safety_threshold]
            y_filtered2 = y_filtered[y_filtered[:, safety_index] < safety_threshold]
        else:
            x_filtered2 = x_filtered[y_filtered[:, safety_index] > safety_threshold]
            y_filtered2 = y_filtered[y_filtered[:, safety_index] > safety_threshold]
        length = x_filtered2.shape[0]
        indexes = np.random.choice(length, n, replace=False)
        return x_filtered2[indexes], y_filtered2[indexes]

    def sample_only_in_small_box_and_safe(
        self, n, box_length, safety_threshold, safety_index, safety_is_upper_bound=True
    ):
        safe_set_found = False
        while not safe_set_found:
            box_left = np.random.uniform(0.0, 1.0 - box_length)
            box_bottom = np.random.uniform(0.0, 1.0 - box_length)
            print(box_left)
            print(box_bottom)
            box_right = box_left + box_length
            box_top = box_bottom + box_length

            x_filtered = self.x[self.x[:, 0] >= box_left]
            y_filtered = self.y[self.x[:, 0] >= box_left]
            x_filtered2 = x_filtered[x_filtered[:, 0] <= box_right]
            y_filtered2 = y_filtered[x_filtered[:, 0] <= box_right]

            x_filtered3 = x_filtered2[x_filtered2[:, 1] <= box_top]
            y_filtered3 = y_filtered2[x_filtered2[:, 1] <= box_top]

            x_filtered4 = x_filtered3[x_filtered3[:, 1] >= box_bottom]
            y_filtered4 = y_filtered3[x_filtered3[:, 1] >= box_bottom]

            noise = np.random.randn(y_filtered4.shape[0], self.d_output) * self.observation_noise
            if self.add_noise:
                y_filtered4 = y_filtered4 + noise

            # y_filtered4 = np.squeeze(y_filtered4)
            if safety_is_upper_bound:
                x_filtered5 = x_filtered4[y_filtered4[:, safety_index] < safety_threshold]
                y_filtered5 = y_filtered4[y_filtered4[:, safety_index] < safety_threshold]
            else:
                x_filtered5 = x_filtered4[y_filtered4[:, safety_index] > safety_threshold]
                y_filtered5 = y_filtered4[y_filtered4[:, safety_index] > safety_threshold]

            length = x_filtered5.shape[0]
            if length >= n:
                indexes = np.random.choice(length, n, replace=False)
                safe_set_found = True

            else:
                print("not engough safe points in box - try new box")

        return x_filtered5[indexes], y_filtered5[indexes]

    def plot(self):
        xs, _, ys, _ = self.sample(700)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(xs[:, 0], xs[:, 1], ys, marker=".")
        plt.show()

    def plot_regime(self, safety_threshold, left=False):
        xs, ys = self.sample_only_one_regime_and_safe(100, safety_threshold, left=left)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(xs[:, 0], xs[:, 1], ys, marker=".")
        plt.show()

    def plot_safe(self, threshold):
        xs, _, ys = self.sample(700)

        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        indexes = np.argwhere(ys[:, 0] < threshold)
        indexes = np.squeeze(indexes)
        x_safe = xs[indexes]
        y_safe = ys[indexes]
        indexes_unsafe = np.argwhere(ys[:, 0] >= threshold)
        indexes_unsafe = np.squeeze(indexes_unsafe)
        x_unsafe = xs[indexes_unsafe]
        y_unsafe = ys[indexes_unsafe]

        x_initial, y_initial = self.sample_only_in_small_box_and_safe(10, 0.3, threshold, 0)

        ax1.scatter(x_initial[:, 0], x_initial[:, 1], y_initial[:, 0], marker="o", color="red")
        ax1.scatter(x_unsafe[:, 0], x_unsafe[:, 1], y_unsafe[:, 0], marker=".", color="grey")
        ax1.scatter(x_safe[:, 0], x_safe[:, 1], y_safe[:, 0], marker=".", color="green")

        ax2.scatter(x_initial[:, 0], x_initial[:, 1], y_initial[:, 1], marker="o", color="red")
        ax2.scatter(x_unsafe[:, 0], x_unsafe[:, 1], y_unsafe[:, 1], marker=".", color="grey")
        ax2.scatter(x_safe[:, 0], x_safe[:, 1], y_safe[:, 1], marker=".", color="green")

        plt.show()
