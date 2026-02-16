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
import matplotlib.pyplot as plt
import gpflow
from alef.utils.utils import create_grid
import random


class KernelDataLoader:
    def __init__(self, kernel, dimensions, observation_noise):
        self.kernel = kernel
        self.dimensions = dimensions
        self.observation_noise = observation_noise
        self.amount_train = 0.7

    def eval(self, X, show_plot):
        n = X.shape[0]
        K = self.kernel(X)
        noise = np.random.normal(0, self.observation_noise, n)
        function_values = np.random.multivariate_normal(np.zeros(n), K, 1)
        y = function_values + noise
        y = y.T
        if show_plot:
            if self.dimensions == 1:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                sorted_indexes = np.argsort(X, axis=0)
                ax.plot(np.squeeze(X[sorted_indexes]), np.squeeze(function_values.T[sorted_indexes]), c="grey")
                plt.show()
            elif self.dimensions == 2:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.tricontourf(
                    np.squeeze(X[:, 0]), np.squeeze(X[:, 1]), np.squeeze(function_values), levels=14, cmap="RdBu_r"
                )
                plt.show()
            else:
                print("-Dimension to high to create a plot")
        return y

    def create_on_grid(self, a, b, n_per_dim, show_plot):
        x_train = create_grid(a, b, n_per_dim, self.dimensions)
        n_train = x_train.shape[0]
        n_test = int((n_train / self.amount_train) * (1 - self.amount_train))
        x_test = np.random.uniform(low=a, high=b, size=(n_test, self.dimensions))
        X = np.concatenate((x_train, x_test))
        n = X.shape[0]
        assert n == n_train + n_test
        K = self.kernel(X)
        noise = np.random.normal(0, self.observation_noise, n)
        function_values = np.random.multivariate_normal(np.zeros(n), K, 1)
        gt_function = function_values.T
        y = function_values + noise
        y = y.T
        y_train = y[:n_train]
        assert y_train.shape[0] == n_train
        y_test = y[n_train:]
        if show_plot:
            if self.dimensions == 1:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                sorted_indexes = np.argsort(X, axis=0)
                ax.plot(np.squeeze(X[sorted_indexes]), np.squeeze(function_values.T[sorted_indexes]), c="grey")
                ax.plot(x_train, y_train, ".", color="red")
                ax.plot(x_test, y_test, ".", color="green")
                plt.show()
            elif self.dimensions == 2:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.tricontourf(
                    np.squeeze(X[:, 0]), np.squeeze(X[:, 1]), np.squeeze(gt_function), levels=14, cmap="RdBu_r"
                )
                plt.show()
            else:
                print("-Dimension to high to create a plot")

        if self.dimensions == 1:
            sorted_indexes = np.argsort(X, axis=0)
            X = np.squeeze(X[sorted_indexes], axis=-1)
            print(X.shape)
            gt_function = np.squeeze(gt_function[sorted_indexes], axis=-1)
        return x_train, y_train, x_test, y_test, X, gt_function, y

    def create(self, a, b, n, show_plot):
        X = np.random.uniform(low=a, high=b, size=(n, self.dimensions))
        K = self.kernel(X)
        noise = np.random.normal(0, self.observation_noise, n)
        function_values = np.random.multivariate_normal(np.zeros(n), K, 1)
        gt_function = function_values.T
        y = function_values + noise
        y = y.T
        x_train, y_train, x_test, y_test = self.train_test_split(X, y)
        if show_plot:
            if self.dimensions == 1:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                sorted_indexes = np.argsort(X, axis=0)
                ax.plot(np.squeeze(X[sorted_indexes]), np.squeeze(function_values.T[sorted_indexes]), c="grey")
                ax.plot(x_train, y_train, ".", color="red")
                ax.plot(x_test, y_test, ".", color="green")
                plt.show()
            elif self.dimensions == 2:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.tricontourf(
                    np.squeeze(X[:, 0]), np.squeeze(X[:, 1]), np.squeeze(gt_function), levels=14, cmap="RdBu_r"
                )
                plt.show()
            else:
                print("-Dimension to high to create a plot")
        return x_train, y_train, x_test, y_test, X, gt_function, y

    def get_dims(self):
        return self.dimensions

    def set_amount_train(self, amount_train):
        self.amount_train = amount_train

    def train_test_split(self, x_complete, y_complete):
        n_complete = x_complete.shape[0]
        indexes = list(range(0, n_complete))
        random.shuffle(indexes)
        num_train = int(n_complete * self.amount_train)
        train_indexes = indexes[:num_train]
        test_indexes = indexes[num_train:]
        x_train = x_complete[train_indexes]
        y_train = y_complete[train_indexes]
        x_test = x_complete[test_indexes]
        y_test = y_complete[test_indexes]
        return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    kernel_data_loader = KernelDataLoader(gpflow.kernels.RBF(lengthscales=0.5, variance=1.5), 1, 0.02)
    X = kernel_data_loader.create(-10, 10, 1000, True)
