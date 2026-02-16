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
from alef.utils.plot_utils import active_learning_nd_plot
from alef.data_sets.base_data_set import BaseDataset
from alef.enums.data_sets_enums import InputPreprocessingType, OutputPreprocessingType
import os


class ClosePI(BaseDataset):
    def __init__(self, base_path, file_name="close_pi.csv"):
        self.file_path = os.path.join(base_path, file_name)
        self.input_preprocessing_type = InputPreprocessingType.MIN_MAX_NORMALIZATION
        self.output_preprocessing_type = OutputPreprocessingType.NORMALIZATION
        self.name = "ClosePi"

    def get_name(self):
        return self.name

    def load_data_set(self):
        df = pd.read_csv(self.file_path, sep=",")
        x_delay = np.expand_dims(df["Delay"].to_numpy(), axis=1)
        x_pilot = np.expand_dims(df["Pilotinjection"].to_numpy(), axis=1)
        x_rail = np.expand_dims(df["Railpressure"].to_numpy(), axis=1)
        x_air = np.expand_dims(df["Airmass"].to_numpy(), axis=1)
        x_boost = np.expand_dims(df["Boostpressure"].to_numpy(), axis=1)
        x_control = np.expand_dims(df["Controlstart"].to_numpy(), axis=1)
        engine_noise = np.expand_dims(df["Noise"].to_numpy(), axis=1)

        if self.input_preprocessing_type == InputPreprocessingType.NORMALIZATION:
            x_delay = (x_delay - np.mean(x_delay)) / np.std(x_delay)
            x_pilot = (x_pilot - np.mean(x_pilot)) / np.std(x_pilot)
            x_rail = (x_rail - np.mean(x_rail)) / np.std(x_rail)
            x_air = (x_air - np.mean(x_air)) / np.std(x_air)
            x_boost = (x_boost - np.mean(x_boost)) / np.std(x_boost)
            x_control = (x_control - np.mean(x_control)) / np.std(x_control)
        elif self.input_preprocessing_type == InputPreprocessingType.MIN_MAX_NORMALIZATION:
            x_delay = (x_delay - np.min(x_delay)) / (np.max(x_delay) - np.min(x_delay))
            x_pilot = (x_pilot - np.min(x_pilot)) / (np.max(x_pilot) - np.min(x_pilot))
            x_rail = (x_rail - np.min(x_rail)) / (np.max(x_rail) - np.min(x_rail))
            x_air = (x_air - np.min(x_air)) / (np.max(x_air) - np.min(x_air))
            x_boost = (x_boost - np.min(x_boost)) / (np.max(x_boost) - np.min(x_boost))
            x_control = (x_control - np.min(x_control)) / (np.max(x_control) - np.min(x_control))

        if self.output_preprocessing_type == OutputPreprocessingType.NORMALIZATION:
            engine_noise = (engine_noise - np.mean(engine_noise)) / np.std(engine_noise)

        self.x = np.concatenate((x_delay, x_pilot, x_rail, x_air, x_boost, x_control), axis=1)
        self.y = engine_noise
        self.length = len(self.y)

    def get_complete_dataset(self):
        return self.x, self.y

    def find_closest_datapoint_for_point(self, point):
        norms = np.linalg.norm(self.x - point, axis=1)
        min_norm_index = np.argmin(norms)
        return self.x[min_norm_index], self.y[min_norm_index]

    def sample(self, n, random_x=False, expand_dims=None):
        if random_x:
            upper_bounds = np.max(self.x, axis=0)
            lower_bounds = np.min(self.x, axis=0)
            random_data_list = []
            for index, upper_bound in enumerate(upper_bounds):
                lower_bound = lower_bounds[index]
                random_at_dimension = np.random.uniform(lower_bound, upper_bound, size=(n, 1))
                random_data_list.append(random_at_dimension)
            random_data = np.concatenate(random_data_list, axis=1)
            x_sample = []
            y_sample = []
            for point in random_data:
                clostest_data_point_x, clostest_data_point_y = self.find_closest_datapoint_for_point(point)
                x_sample.append(clostest_data_point_x)
                y_sample.append(clostest_data_point_y)
            x_sample = np.array(x_sample)
            y_sample = np.array(y_sample)
        else:
            if n > self.length:
                n = self.length
            indexes = np.random.choice(self.length, n, replace=False)
            x_sample = self.x[indexes]
            y_sample = self.y[indexes]
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
        x_train = self.x[train_indexes]
        y_train = self.y[train_indexes]
        x_test = self.x[test_indexes]
        y_test = self.y[test_indexes]
        return x_train, y_train, x_test, y_test

    def get_close_samples(self, n, only_safe=False, safety_threshold=0):
        reference_index = np.random.choice(self.length, 1)
        reference_point = self.x[reference_index]
        norms = np.linalg.norm(self.x - reference_point, axis=1)
        indexes_smallest_norms = np.argsort(norms)[: n * 5]
        x_sample = self.x[indexes_smallest_norms]
        y_sample = self.y[indexes_smallest_norms]
        indexes = np.random.choice(n * 5, n, replace=False)
        x_sample = x_sample[indexes]
        y_sample = y_sample[indexes]

    def plot(self, index):
        xs, ys = self.sample(3000)
        fig, ax = plt.subplots()
        ax.plot(xs[:, index], ys, ".")
        plt.show()

    def hist(self, index, n, random_x=True):
        xs, _ = self.sample(n, random_x=random_x)
        fig, ax = plt.subplots()
        ax.hist(xs[:, index], bins=40)
        plt.show()

    def plot_scatter(self):
        xs, ys = self.sample(300)
        active_learning_nd_plot(xs, ys)
