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
import tensorflow as tf
from copy import deepcopy
from tensorflow_probability import distributions as tfd
from gpflow.utilities import print_summary

from alef.enums.environment_enums import GPFramework
from alef.gp_samplers.base_distribution import BaseDistribution
from alef.kernels.kernel_factory import KernelFactory

from alef.configs.base_parameters import NOISE_VARIANCE_LOWER_BOUND
from alef.configs.prior_parameters import EXPECTED_OBSERVATION_NOISE
from alef.configs.kernels.base_kernel_config import BaseKernelConfig


class GPDistribution(BaseDistribution):
    def __init__(
        self,
        kernel_config: BaseKernelConfig,
        observation_noise: float,
        expected_observation_noise: float = EXPECTED_OBSERVATION_NOISE,
    ):
        r"""
        :param kernel_config: BaseKernelConfig
        :param observation_noise: float, observation noise standard deviation
        :param expected_observation_noise: float, mean of noise prior (noise in standard deviation)
        """
        super().__init__(GPFramework.GPFLOW)
        self._original_kernel = KernelFactory.build(kernel_config)
        self.kernel_list = [deepcopy(self._original_kernel)]
        self.noise_variance = self._original_noise_variance = np.power(observation_noise, 2.0)
        self.noise_prior = tfd.Exponential(1 / np.power(expected_observation_noise, 2.0))

    def draw_parameter(self, num_priors: int = 1, num_functions: int = 1, draw_hyper_prior: bool = False):
        """
        draw hyper-priors, f, or noise of y|f

        arguments:

        num_priors: batch size of kernel hyperparameters (i.e. num of kernels).
        num_functions: number of functional sample given a GP prior.
        draw_hyper_prior: whether to draw parameters from hyper-priors.

        """
        assert num_priors > 0
        assert num_functions > 0
        self._num_functions = num_functions

        if num_priors > 1:
            self.kernel_list = [deepcopy(self._original_kernel) for _ in range(num_priors)]
            self.noise_variance = np.tile(self._original_noise_variance, (num_priors,))
        else:  # reset to original
            self.kernel = deepcopy(self._original_kernel)
            self.noise_variance = deepcopy(self._original_noise_variance)

        if draw_hyper_prior:
            for kernel in self.kernel_list:
                for parameter in kernel.trainable_parameters:
                    new_value = parameter.prior.sample()
                    parameter.assign(new_value)

            if hasattr(self.noise_variance, "__len__"):
                self.noise_variance = self.noise_prior.sample(num_priors).numpy() + NOISE_VARIANCE_LOWER_BOUND
            else:
                self.noise_variance = self.noise_prior.sample().numpy() + NOISE_VARIANCE_LOWER_BOUND

    def show_parameter(self):
        for k in self.kernel_list:
            print_summary(k.kernel)
        print(
            f"observation noise variance: {self.noise_variance}, type {self.noise_variance.dtype}, prior {self.noise_prior}"
        )

    def f_sampler(self, x_data: tf.Tensor):
        """
        compute GP( mean(x_data), kernel(x_data) ), return in raw tf type

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        """
        assert len(x_data.shape) in [2, 4]
        N = tf.shape(x_data)[-2]
        if len(x_data.shape) == 4:
            K = tf.concat(
                [
                    tf.concat(
                        [k(x_data[i, j, ...], full_cov=True)[None, ...] for j in range(self._num_functions)], axis=0
                    )[None, ...]
                    + NOISE_VARIANCE_LOWER_BOUND * tf.eye(N, dtype=x_data.dtype)
                    for i, k in enumerate(self.kernel_list)
                ],
                axis=0,
            )
        else:
            K = self.kernel_list[0](x_data, full_cov=True) + NOISE_VARIANCE_LOWER_BOUND * tf.eye(N, dtype=x_data.dtype)

        L = tf.linalg.cholesky(K)

        return tfd.MultivariateNormalTriL(scale_tril=L)

    def y_sampler(self, x_data: tf.Tensor):
        """
        compute GP( mean(x_data), kernel(x_data) ) + noise_dist, return in raw tf or torch type

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        """
        assert len(x_data.shape) in [2, 4]
        N = tf.shape(x_data)[-2]
        if len(x_data.shape) == 4:
            K = tf.concat(
                [
                    tf.concat(
                        [k(x_data[i, j, ...], full_cov=True)[None, ...] for j in range(self._num_functions)], axis=0
                    )[None, ...]
                    + self.noise_variance[i] * tf.eye(N, dtype=x_data.dtype)
                    for i, k in enumerate(self.kernel_list)
                ],
                axis=0,
            )
        else:
            K = self.kernel_list[0](x_data, full_cov=True) + self.noise_variance * tf.eye(N, dtype=x_data.dtype)

        L = tf.linalg.cholesky(K)
        return tfd.MultivariateNormalTriL(scale_tril=L)

    def sample_f(self, x_data: np.ndarray):
        """
        sample f from GP( mean(x_data), kernel(x_data) )

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        """
        return self.f_sampler(x_data).sample().numpy()

    def sample_y(self, x_data: np.ndarray):
        """
        sample y from GP( mean(x_data), kernel(x_data) ) + noise_dist(x_data)

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        """
        return self.y_sampler(x_data).sample().numpy()
