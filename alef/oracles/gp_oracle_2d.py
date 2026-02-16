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

import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from gpflow.utilities import to_default_float
from alef.configs.kernels.base_elementary_kernel_config import BaseElementaryKernelConfig
from alef.configs.kernels.pytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig
from alef.gp_samplers.gp_gpflow_distribution import GPDistribution
from alef.gp_samplers.gp_gpytorch_distribution import GPTorchDistribution
from alef.utils.utils import create_grid

f64 = to_default_float


class GPOracle2D:
    def __init__(self, kernel_config, noise_level):
        self.noise_level = noise_level
        self.__dimension = 2
        assert kernel_config.input_dimension == 2
        if isinstance(kernel_config, BaseElementaryKernelConfig):
            self.gp_dist = GPDistribution(kernel_config, noise_level)
        elif isinstance(kernel_config, BaseKernelPytorchConfig):
            self.gp_dist = GPTorchDistribution(kernel_config, noise_level)

    def initialize(self, a, b, n):
        self.__a = a
        self.__b = b
        grid = create_grid(self.__a, self.__b, n, self.__dimension)
        self.grid = grid
        function_values = np.squeeze(self.gp_dist.sample_f(grid))
        self.function_values = function_values
        self.f = interpolate.interp2d(grid[:, 0], grid[:, 1], function_values, kind="linear")

    def draw_from_hyperparameter_prior(self):
        print("-Draw from hyperparameter prior")
        self.gp_dist.draw_parameter(draw_hyper_prior=True)
        self.gp_dist.show_parameter()

    def query(self, x, noisy=True):
        function_value = np.squeeze(self.f(x[0], x[1]))
        if noisy:
            epsilon = np.random.normal(0, self.noise_level, 1)[0]
            function_value += epsilon
        return function_value

    def get_random_data(self, n, noisy=True):
        X = np.random.uniform(low=self.__a, high=self.__b, size=(n, self.__dimension))
        function_values = []
        for x in X:
            function_value = self.query(x, noisy)
            function_values.append(function_value)
        return X, np.expand_dims(np.array(function_values), axis=1)

    def get_box_bounds(self):
        return self.__a, self.__b

    def get_dimension(self):
        return self.__dimension

    def plot(self, save_fig=False, path=None, file_name=None):
        xs = self.grid
        ys = self.function_values
        # xs, ys = self.get_random_data(300, noisy=False)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        ax.plot_trisurf(np.squeeze(xs[:, 0]), np.squeeze(xs[:, 1]), np.squeeze(ys), linewidth=1.2, cmap="viridis")
        # ax.scatter(xs[:, 0], xs[:, 1], ys, marker=".")
        if save_fig:
            plt.savefig(os.path.join(path, file_name))
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    from alef.configs.kernels.rbf_configs import RBFWithPriorConfig

    for i in range(0, 10):
        gp_oracle = GPOracle2D(RBFWithPriorConfig(input_dimension=2), 0.01)
        gp_oracle.draw_from_hyperparameter_prior()
        gp_oracle.initialize(0, 1, 50)

        gp_oracle.plot(False, None, "main.png")
