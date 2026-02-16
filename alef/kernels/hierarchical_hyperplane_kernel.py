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

import gpflow
from gpflow.utilities import positive
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np

from alef.kernels.input_initialized_kernel_interface import InputInitializedKernelInterface
from typing import Tuple

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)

f64 = gpflow.utilities.to_default_float


class HierarchicalHyperplaneKernel(gpflow.kernels.Kernel):
    def __init__(
        self,
        base_kernel: gpflow.kernels.Kernel,
        base_hyperplane_mu: float,
        base_hyperplane_std: float,
        input_dimension: int,
        base_smoothing: float,
        hyperplanes_learnable: bool,
        learn_smoothing_parameter: bool,
        topology: int,
        smoothing_prior_parameters: Tuple[float, float],
        **kwargs,
    ):
        super().__init__()
        self.topology = topology
        self.set_default_topology(topology)
        self.dimension = input_dimension
        self.learn_smoothing_parameter = learn_smoothing_parameter
        self.smoothing_prior_parameters = smoothing_prior_parameters
        self.hyperplanes_learnable = hyperplanes_learnable
        self.base_hyperplane_mu = base_hyperplane_mu
        self.base_hyperplane_std = base_hyperplane_std
        self.base_smoothing = base_smoothing
        self.kernel_list = []
        for i in range(0, self.n_experts):
            kernel = gpflow.utilities.deepcopy(base_kernel)
            self.kernel_list.append(kernel)
        self.initialize_hyperplane_parameters()

    def initialize_hyperplane_parameters(self):
        self.smoothing_list = []
        self.hyperplane_parameter_list = []
        for j in range(0, self.M):
            smoothing_param = gpflow.Parameter(
                [self.base_smoothing], transform=positive(), trainable=self.learn_smoothing_parameter
            )
            a_smoothing, b_smoothing = self.smoothing_prior_parameters
            smoothing_param.prior = tfd.Gamma(f64([a_smoothing]), f64([b_smoothing]))
            self.smoothing_list.append(smoothing_param)
            w = gpflow.Parameter(
                np.repeat(self.base_hyperplane_mu, self.dimension + 1), trainable=self.hyperplanes_learnable
            )
            w.prior = tfd.Normal(
                np.repeat(self.base_hyperplane_mu, self.dimension + 1),
                np.repeat(self.base_hyperplane_std, self.dimension + 1),
            )
            self.hyperplane_parameter_list.append(w)

    def set_kernel_parameters(self, lengthscales_list, variances_list, smoothing_value_list):
        assert len(lengthscales_list) == self.n_experts
        for i in range(0, self.n_experts):
            self.kernel_list[i].lengthscales.assign(lengthscales_list[i])
            self.kernel_list[i].variance.assign(variances_list[i])
        for j in range(0, self.M):
            self.smoothing_list[j].assign(smoothing_value_list[j])

    def set_smoothing_list(self, smoothing_value_list):
        for j in range(0, self.M):
            self.smoothing_list[j].assign(smoothing_value_list[j])

    def set_hyperplane_parameters(self, hyperplane_value_list):
        for j in range(0, self.M):
            self.hyperplane_parameter_list[j].assign(hyperplane_value_list[j])

    def gate(self, x):
        expert_probabilities = []
        for k in range(0, self.n_experts):
            prob = tf.cast(tf.expand_dims(tf.repeat(1.0, x.shape[0]), axis=1), dtype=tf.float64)
            for j in range(0, self.M):
                w = self.hyperplane_parameter_list[j]
                w_0 = w[0]
                w_rest = tf.expand_dims(w[1:], axis=0)
                smoothing = self.smoothing_list[j][0]
                prob_elem = tf.math.pow(
                    self.sigmoid(w_0 + tf.linalg.matmul(x, tf.transpose(w_rest)), smoothing), self.left_matrix[k, j]
                ) * tf.math.pow(
                    1 - self.sigmoid(w_0 + tf.linalg.matmul(x, tf.transpose(w_rest)), smoothing),
                    self.right_matrix[k, j],
                )
                prob = prob * prob_elem
            expert_probabilities.append(prob)
        return expert_probabilities

    def fast_gate(self, x):
        pass

    def sigmoid(self, x, smoothing):
        return 1 / (1 + tf.math.exp(-1 * x * smoothing))

    def set_default_topology(self, n_depth):
        if n_depth == 1:
            left_matrix = np.array([[1], [0]])
            right_matrix = np.array([[0], [1]])
            self.set_topology_matrices(left_matrix, right_matrix)
        elif n_depth == 2:
            left_matrix = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]])
            right_matrix = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1]])
            self.set_topology_matrices(left_matrix, right_matrix)
        elif n_depth == 3:
            left_matrix = np.array(
                [
                    [1, 1, 0, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            )
            right_matrix = np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0],
                    [1, 0, 1, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 1],
                ]
            )
            self.set_topology_matrices(left_matrix, right_matrix)

    def set_topology_matrices(self, left_matrix, right_matrix):
        self.left_matrix = left_matrix
        self.right_matrix = right_matrix
        self.n_experts = self.left_matrix.shape[0]  ## Number of experts
        self.M = self.left_matrix.shape[1]  ## Number of gate nodes
        assert self.n_experts == self.right_matrix.shape[0]
        assert (self.left_matrix.shape[0] * self.left_matrix.shape[1]) == (
            (self.left_matrix == 0).sum() + (self.left_matrix == 1).sum()
        )
        assert (self.right_matrix.shape[0] * self.right_matrix.shape[1]) == (
            (self.right_matrix == 0).sum() + (self.right_matrix == 1).sum()
        )

    def get_topology(self):
        return self.topology

    def K(self, X, X2=None):
        len_x = X.shape[0]
        if X2 is None:
            X2 = X
            len_x2 = len_x
        else:
            len_x2 = X2.shape[0]
        gate_x = self.gate(X)
        gate_x2 = self.gate(X2)
        output = tf.zeros((len_x, len_x2), dtype=tf.dtypes.float64)
        for k in range(0, self.n_experts):
            output += tf.matmul(gate_x[k], tf.transpose(gate_x2[k])) * self.kernel_list[k].K(X, X2)
        return output

    def K_diag(self, X):
        len_x = X.shape[0]
        gate_x = self.gate(X)
        output = tf.zeros((len_x,), dtype=tf.dtypes.float64)
        gate_x = self.gate(X)
        for k in range(0, self.n_experts):
            output += tf.math.multiply(tf.squeeze(tf.math.pow(gate_x[k], 2.0)), self.kernel_list[k].K_diag(X))
        return output


def get_num_active_partitions(expert_probabilities: np.array, threshold: float):
    num_active_partitions = 0
    for expert_prob in expert_probabilities:
        if np.any(np.greater(expert_prob, threshold)):
            num_active_partitions += 1
    return num_active_partitions


class HKKInputInitializedBaseKernel(HierarchicalHyperplaneKernel, InputInitializedKernelInterface):
    def __init__(
        self,
        base_kernel: gpflow.kernels.Kernel,
        base_hyperplane_mu: float,
        base_hyperplane_std: float,
        input_dimension: int,
        base_smoothing: float,
        hyperplanes_learnable: bool,
        learn_smoothing_parameter: bool,
        topology: int,
        smoothing_prior_parameters: Tuple[float, float],
        **kwargs,
    ):
        assert isinstance(base_kernel, InputInitializedKernelInterface)
        super().__init__(
            base_kernel,
            base_hyperplane_mu,
            base_hyperplane_std,
            input_dimension,
            base_smoothing,
            hyperplanes_learnable,
            learn_smoothing_parameter,
            topology,
            smoothing_prior_parameters,
            **kwargs,
        )

    def initialize_parameters(self, x_data, y_data):
        self.initialize_hyperplane_parameters()
        for kernel in self.kernel_list:
            assert isinstance(kernel, InputInitializedKernelInterface)
            kernel.initialize_parameters(x_data, y_data)


if __name__ == "__main__":
    kernel = HierarchicalHyperplaneKernel(
        hyperplane_list_mus=[np.array([10.0, -10.0, 0.0])], hyperplane_list_sds=[np.array([0.2, 0.2, 0.2])]
    )
    print(kernel.gate(np.array([[1.1, 0.5], [0.3, 0.5]])))
    print(kernel.K(np.array([[1.1, 0.5], [0.3, 0.5], [2.0, 1.0]])))
