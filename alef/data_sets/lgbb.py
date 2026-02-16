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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from alef.data_sets.base_data_set import BaseDataset
import os


class OutputType(Enum):
    LIFT = 0
    ROLL = 1
    YAW = 2
    PITCH = 3
    DRAG = 4


class LGBB(BaseDataset):
    def __init__(
        self,
        base_path,
        file_name="lgbb_corrected.txt",
        observation_noise=0.01,
        normalize_output=True,
    ):
        self.file_path = os.path.join(base_path, file_name)
        self.observation_noise = observation_noise
        self.add_noise = True
        self.exclude_outlier = True
        self.output_type = OutputType.LIFT
        self.beta = 0.0
        self.normalize_output = normalize_output
        self.name = "LGBB"

    def get_name(self):
        return self.name

    def load_data_set(self):
        df = pd.read_csv(self.file_path, sep=" ")
        df_beta_0 = df[df["beta"] == self.beta]

        if self.output_type == OutputType.LIFT:
            x1 = df_beta_0["mach"].to_numpy() / 6
            x2 = (df_beta_0["alpha"].to_numpy() + 5) / 35
            self.y = np.expand_dims(df_beta_0["lift"].to_numpy(), axis=1)
        elif self.output_type == OutputType.ROLL:
            x1 = df_beta_0["mach"].to_numpy() / 6
            x2 = (df_beta_0["alpha"].to_numpy() + 5) / 35
            self.y = np.expand_dims(df_beta_0["roll"].to_numpy(), axis=1)
        elif self.output_type == OutputType.YAW:
            key = "yaw"
            low = df_beta_0[key].quantile(0.01)
            high = df_beta_0[key].quantile(0.99)
            df_filtered = df_beta_0[(df_beta_0[key] < high) & (df_beta_0[key] > low)]
            x1 = df_filtered["mach"].to_numpy() / 6
            x2 = (df_filtered["alpha"].to_numpy() + 5) / 35
            self.y = np.expand_dims(df_filtered[key].to_numpy(), axis=1)
        elif self.output_type == OutputType.PITCH:
            key = "pitch"
            low = df_beta_0[key].quantile(0.01)
            high = df_beta_0[key].quantile(0.99)
            df_filtered = df_beta_0[(df_beta_0[key] < high) & (df_beta_0[key] > low)]
            x1 = df_filtered["mach"].to_numpy() / 6
            x2 = (df_filtered["alpha"].to_numpy() + 5) / 35
            self.y = np.expand_dims(df_filtered[key].to_numpy(), axis=1)
        elif self.output_type == OutputType.DRAG:
            key = "drag"
            low = df_beta_0[key].quantile(0.01)
            high = df_beta_0[key].quantile(0.99)
            df_filtered = df_beta_0[(df_beta_0[key] < high) & (df_beta_0[key] > low)]
            x1 = df_filtered["mach"].to_numpy() / 6
            x2 = (df_filtered["alpha"].to_numpy() + 5) / 35
            self.y = np.expand_dims(df_filtered[key].to_numpy(), axis=1)

        if self.normalize_output:
            mean_y = np.mean(self.y)
            std_y = np.std(self.y)
            self.y = (self.y - mean_y) / std_y

        self.x = np.stack((x1, x2), axis=1)
        self.length = x1.shape[0]

    def get_complete_dataset(self, add_noise=True):
        n = self.y.shape[0]
        if add_noise:
            noise = np.random.randn(n, 1) * self.observation_noise
            y = self.y + noise
        else:
            y = self.y
        return self.x, y

    def sample(self, n, random_x=None, expand_dims=None):
        indexes = np.random.choice(self.length, n, replace=False)
        x_sample = self.x[indexes]
        f_sample = self.y[indexes]
        noise = np.random.randn(n, 1) * self.observation_noise
        if self.add_noise:
            y_sample = f_sample + noise
        else:
            y_sample = f_sample
        return x_sample, y_sample

    def sample_train_test(self, use_absolute: bool, n_train: int, n_test: int, fraction_train: float):
        if use_absolute:
            assert n_train < self.length
            n = n_train + n_test
            if n > self.length:
                n = self.length
                print("Test + Train set exceeds number of datapoints - use n-n_train test points")
        else:
            n = self.length
            n_train = int(fraction_train * n)
            n_test = n - n_train
        indexes = np.random.choice(self.length, n, replace=False)
        train_indexes = indexes[:n_train]
        assert len(train_indexes) == n_train
        test_indexes = indexes[n_train:]
        if use_absolute and n_train + n_test <= self.length:
            assert len(test_indexes) == n_test
        noise = np.random.randn(self.length, 1) * self.observation_noise
        if self.add_noise:
            self.y + noise
        else:
            self.y
        x_train = self.x[train_indexes]
        y_train = self.y[train_indexes]
        x_test = self.x[test_indexes]
        y_test = self.y[test_indexes]
        return x_train, y_train, x_test, y_test

    def sample_with_cats(self, n, random_x=None, expand_dims=None):
        indexes = np.random.choice(self.length, n, replace=False)
        x_sample = self.x[indexes]
        f_sample = self.y[indexes]
        noise = np.random.randn(n, 1) * self.observation_noise
        if self.add_noise:
            y_sample = f_sample + noise
        else:
            y_sample = f_sample
        cat_indexes = np.argwhere(x_sample[:, 0] >= 1.0)
        cats_sample = np.repeat(1.0, x_sample[:, 0].shape[0])
        cats_sample[cat_indexes] = 0.0
        return x_sample, f_sample, y_sample, cats_sample

    def sample_only_one_regime_and_safe(
        self, n, safety_threshold, left=False, x1_threshold=0.2, safety_is_upper_bound=False
    ):
        if left:
            x_filtered = self.x[self.x[:, 0] < x1_threshold]
            y_filtered = self.y[self.x[:, 0] < x1_threshold]
        else:
            x_filtered = self.x[self.x[:, 0] >= x1_threshold]
            y_filtered = self.y[self.x[:, 0] >= x1_threshold]
        y_filtered = np.squeeze(y_filtered)
        if safety_is_upper_bound:
            x_filtered2 = x_filtered[y_filtered < safety_threshold]
            y_filtered2 = y_filtered[y_filtered < safety_threshold]
        else:
            x_filtered2 = x_filtered[y_filtered > safety_threshold]
            y_filtered2 = y_filtered[y_filtered > safety_threshold]
        length = x_filtered2.shape[0]
        indexes = np.random.choice(length, n, replace=False)
        return x_filtered2[indexes], np.expand_dims(y_filtered2[indexes], axis=1)

    def sample_only_in_small_box_and_safe(self, n, box_length, safety_threshold, safety_is_upper_bound=True):
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

            noise = np.random.randn(y_filtered4.shape[0], 1) * self.observation_noise
            if self.add_noise:
                y_filtered4 = y_filtered4 + noise

            y_filtered4 = np.squeeze(y_filtered4)
            if safety_is_upper_bound:
                x_filtered5 = x_filtered4[y_filtered4 < safety_threshold]
                y_filtered5 = y_filtered4[y_filtered4 < safety_threshold]
            else:
                x_filtered5 = x_filtered4[y_filtered4 > safety_threshold]
                y_filtered5 = y_filtered4[y_filtered4 > safety_threshold]

            length = x_filtered5.shape[0]
            if length >= n:
                indexes = np.random.choice(length, n, replace=False)
                safe_set_found = True

            else:
                print("not engough safe points in box - try new box")

        return x_filtered5[indexes], np.expand_dims(y_filtered5[indexes], axis=1)

    def plot(self):
        xs, ys = self.sample(700)
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
        xs, ys = self.sample(700)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        indexes = np.argwhere(np.squeeze(ys) < threshold)
        indexes = np.squeeze(indexes)
        x_safe = xs[indexes]
        y_safe = ys[indexes]
        indexes_unsafe = np.argwhere(np.squeeze(ys) >= threshold)
        indexes_unsafe = np.squeeze(indexes_unsafe)
        x_unsafe = xs[indexes_unsafe]
        y_unsafe = ys[indexes_unsafe]

        x_initial, y_initial = self.sample_only_in_small_box_and_safe(10, 0.3, threshold)
        ax.scatter(x_initial[:, 0], x_initial[:, 1], y_initial, marker="o", color="red")
        ax.scatter(x_unsafe[:, 0], x_unsafe[:, 1], y_unsafe, marker=".", color="grey")
        ax.scatter(x_safe[:, 0], x_safe[:, 1], y_safe, marker=".", color="green")

        plt.show()
