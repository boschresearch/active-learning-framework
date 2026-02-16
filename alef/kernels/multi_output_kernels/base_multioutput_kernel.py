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
import tensorflow as tf
from typing import Dict
from .latent_kernel_enum import LatentKernel


class BaseMultioutputKernel(gpflow.kernels.MultioutputKernel):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.active_dimension = active_dimension
        self.active_on_single_dimension = active_on_single_dimension
        if active_on_single_dimension:
            super().__init__(name=name + "_on_" + str(active_dimension))
            self.num_active_dimensions = 1
        else:
            super().__init__(name=name)
            self.num_active_dimensions = input_dimension
        self.kernel = None

    @property
    def num_latent_gps(self):
        """The number of latent GPs in the multioutput kernel"""
        return self.kernel.num_latent_gps

    @property
    def latent_kernels(self):
        """The underlying kernels in the multioutput kernel"""
        return self.kernel.latent_kernels

    def K(self, X, X2=None, full_output_cov=True):
        """
        Returns the correlation of f(X) and f(X2), where f(.) can be multi-dimensional.
        :param X: data matrix, [N1, D]
        :param X2: data matrix, [N2, D]
        :param full_output_cov: calculate correlation between outputs.
        :return: cov[f(X), f(X2)] with shape
        - [N1, P, N2, P] if `full_output_cov` = True
        - [P, N1, N2] if `full_output_cov` = False
        """
        if X2 is None:
            X2 = X
        if self.active_on_single_dimension:
            X = tf.expand_dims(X[:, self.active_dimension], axis=1)
            X2 = tf.expand_dims(X2[:, self.active_dimension], axis=1)
        assert X.shape[1] == self.num_active_dimensions
        return self.kernel.K(X, X2, full_output_cov=full_output_cov)

    def K_diag(self, X, full_output_cov=True):
        """
        Returns the correlation of f(X) and f(X), where f(.) can be multi-dimensional.
        :param X: data matrix, [N, D]
        :param full_output_cov: calculate correlation between outputs.
        :return: var[f(X)] with shape
        - [N, P, N, P] if `full_output_cov` = True
        - [N, P] if `full_output_cov` = False
        """
        if self.active_on_single_dimension:
            X = tf.expand_dims(X[:, self.active_dimension], axis=1)
        return self.kernel.K_diag(X, full_output_cov=full_output_cov)

    def get_input_dimension(self):
        return self.input_dimension

    def get_output_dimension(self):
        return self.output_dimension

    def assign_parameters(self, parameter_values: Dict):
        def get_attribute(class_obj, name):
            if "[" in name:
                key, idx = name.split("]")[0].split("[")
                return getattr(class_obj, key)[int(idx)]
            else:
                return getattr(class_obj, name)

        for key, values in parameter_values.items():
            target = self.kernel
            if "." in key:
                for name in key.split("."):
                    target = get_attribute(target, name)
            else:
                target = get_attribute(target, key)
            target.assign(values)

    def pick_kernel_object(self, latent_kernel: LatentKernel):
        if latent_kernel == LatentKernel.RBF:
            return gpflow.kernels.RBF
        elif latent_kernel == LatentKernel.MATERN12:
            return gpflow.kernels.Matern12
        elif latent_kernel == LatentKernel.MATERN32:
            return gpflow.kernels.Matern32
        elif latent_kernel == LatentKernel.MATERN52:
            return gpflow.kernels.Matern52
        else:
            raise NotImplementedError
