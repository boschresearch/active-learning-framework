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
from alef.utils.plotter import Plotter
from scipy import interpolate
from gpflow.utilities import to_default_float
from alef.configs.kernels.base_elementary_kernel_config import BaseElementaryKernelConfig
from alef.configs.kernels.pytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig
from alef.gp_samplers.gp_gpflow_distribution import GPDistribution
from alef.gp_samplers.gp_gpytorch_distribution import GPTorchDistribution

f64 = to_default_float


class GPOracle1D:
    def __init__(self, kernel_config, noise_level):
        self.noise_level = noise_level
        self.__dimension = 1
        assert kernel_config.input_dimension == 1
        if isinstance(kernel_config, BaseElementaryKernelConfig):
            self.gp_dist = GPDistribution(kernel_config, noise_level)
        elif isinstance(kernel_config, BaseKernelPytorchConfig):
            self.gp_dist = GPTorchDistribution(kernel_config, noise_level)

    def initialize(self, a, b, n):
        self.__a = a
        self.__b = b
        grid = np.linspace(self.__a, self.__b, n)
        function_values = np.squeeze(self.gp_dist.sample_f(np.expand_dims(grid, axis=1)))
        self.f = interpolate.interp1d(grid, function_values, kind="linear")

    def draw_from_hyperparameter_prior(self):
        print("-Draw from hyperparameter prior")
        self.gp_dist.draw_parameter(draw_hyper_prior=True)
        self.gp_dist.show_parameter()

    def query(self, x, noisy=True):
        function_value = np.squeeze(self.f(x))
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
        print("-Dataset of length " + str(n) + " generated")
        return X, np.expand_dims(np.array(function_values), axis=1)

    def get_random_data_in_interval(self, n, a, b, noisy=True):
        X = np.random.uniform(low=a, high=b, size=(n, self.__dimension))
        function_values = []
        for x in X:
            function_value = self.query(x, noisy)
            function_values.append(function_value)
        print("-Dataset of length " + str(n) + " generated")
        return X, np.expand_dims(np.array(function_values), axis=1)

    def get_box_bounds(self):
        return self.__a, self.__b

    def get_dimension(self):
        return self.__dimension

    def plot(self):
        X, y = self.get_random_data(1000, False)
        plotter_object = Plotter(1)
        plotter_object.add_gt_function(np.squeeze(X), np.squeeze(y), "blue", 0)
        plotter_object.show()


if __name__ == "__main__":
    from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import RBFWithPriorPytorchConfig

    for i in range(0, 10):
        # kernel_config = RBFWithPriorConfig(input_dimension=1)
        kernel_config = RBFWithPriorPytorchConfig(input_dimension=1)

        gpOracle = GPOracle1D(kernel_config, 0.1)

        gpOracle.draw_from_hyperparameter_prior()
        gpOracle.initialize(0, 1, 1000)
        gpOracle.plot()
