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
import numpy as np
import tensorflow as tf
from typing import Optional, Dict
from .latent_kernel_enum import LatentKernel


class BaseMultioutputFlattenedKernel(gpflow.kernels.Kernel):
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

    def __call__(self, X, X2: Optional = None, *, full_cov: bool = True, presliced: bool = False):
        if self.active_on_single_dimension:
            D = self.get_input_dimension()
            X = tf.gather(X, [self.active_dimension, D], axis=-1)
            if not X2 is None:
                X2 = tf.gather(X2, [self.active_dimension, D], axis=-1)

        return self.kernel(X, X2, full_cov=full_cov, presliced=presliced)

    @property
    def num_latent_gps(self):
        """The number of latent GPs in the multioutput kernel"""
        raise NotImplementedError

    @property
    def latent_kernels(self):
        """The underlying kernels in the multioutput kernel"""
        raise NotImplementedError

    def K(self, X, X2=None):
        """
        gpflow.kernels.Kernel.__call__(**) perform 'slice'.
        Sometimes the kernel has multiple gpflow kernel objects.
        If different kernel use different input dims (active_dims), slice becomes important.
        E.g. gpflow.kernels.Coregion(..., active_dims=[D]) * gpflow.kernels.Matern52(active_dims=tf.range(D))
             might not work properly.

        Please only write a new K when you are absolutely sure about what you are doing.
        """
        raise EnvironmentError("Please do not use this method, see self.__call__(**)")

    def K_diag(self, X):
        """
        gpflow.kernels.Kernel.__call__(**) perform 'slice'.
        Sometimes the kernel has multiple gpflow kernel objects.
        If different kernel use different input dims (active_dims), slice becomes important.
        E.g. gpflow.kernels.Coregion(..., active_dims=[D]) * gpflow.kernels.Matern52(active_dims=tf.range(D))
             might not work properly.

        Please only write a new K_diag when you are absolutely sure about what you are doing.
        """
        raise EnvironmentError("Please do not use this method, see self.__call__(**)")

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

    @property
    def prior_scale(self):
        D = self.get_input_dimension()
        P = self.get_output_dimension()
        dumpy_point = np.zeros([1, D])
        dumpy_point = np.hstack((dumpy_point, np.array([[P - 1]])))

        var_scale = self(dumpy_point, full_cov=False).numpy().reshape(-1)[0]
        std_scale = np.sqrt(var_scale)
        return std_scale

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
