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
import gpflow
import tensorflow as tf
from typing import Tuple, List, Sequence
from gpflow.utilities import set_trainable

from tensorflow_probability import distributions as tfd

from alef.kernels.multi_output_kernels.base_transfer_kernel import BaseTransferKernel
from alef.kernels.multi_output_kernels.latent_kernel_enum import LatentKernel

gpflow.config.set_default_float(np.float64)
f64 = gpflow.utilities.to_default_float


class FilterKernel(gpflow.kernels.Kernel):
    def __init__(
        self,
        output_dim: int,
        pass_index: Sequence,
        *,
        active_dims=None,
        name=None,
    ):
        super().__init__(active_dims=active_dims, name=name)
        self.output_dim = output_dim
        assert len(pass_index) == 2

        mask = np.zeros([self.output_dim, self.output_dim], dtype=bool)
        mask[pass_index[0], pass_index[1]] = True
        mask[pass_index[1], pass_index[0]] = True

        self.B = tf.where(
            mask, f64(tf.ones([self.output_dim, self.output_dim])), f64(tf.zeros([self.output_dim, self.output_dim]))
        )

    @property
    def B_diag(self):
        return tf.linalg.diag_part(self.B)

    def K(self, X, X2=None):
        shape_constraints = [
            (X, [..., "N", 1]),
        ]
        if X2 is not None:
            shape_constraints.append((X2, [..., "M", 1]))
        tf.debugging.assert_shapes(shape_constraints)

        X = tf.cast(X[..., 0], tf.int32)
        if X2 is None:
            X2 = X
        else:
            X2 = tf.cast(X2[..., 0], tf.int32)

        B = self.B

        return tf.gather(tf.transpose(tf.gather(B, X2)), X)

    def K_diag(self, X):
        tf.debugging.assert_shapes([(X, [..., "N", 1])])
        X = tf.cast(X[..., 0], tf.int32)
        B_diag = self.B_diag
        return tf.gather(B_diag, X)


class FlexibleTransferKernel(BaseTransferKernel):
    """
    X: [N, D+1], the last column is binary
    D: input_dimension
    """

    def __init__(
        self,
        variance_list: List,
        lengthscale_list: List,
        input_dimension: int,
        output_dimension: int,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        latent_kernel: LatentKernel,
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        P = output_dimension
        assert P == 2  # only support 1 source right now
        assert len(variance_list) == 3
        assert len(lengthscale_list) == 3
        super().__init__(
            input_dimension,
            P,
            active_on_single_dimension,
            active_dimension,
            name,
            **kwargs,
        )

        var_s = variance_list[0]
        var_st = variance_list[1]
        var_t = variance_list[2]

        leng_s = lengthscale_list[0]
        leng_st = lengthscale_list[1]
        leng_t = lengthscale_list[2]

        def get_lg(leng):
            if hasattr(leng, "__len__"):
                if len(leng) == 1:
                    lg = f64(np.repeat(leng[0], self.num_active_dimensions))
                else:
                    assert len(leng) == self.num_active_dimensions
                    lg = f64(leng)
            else:
                lg = f64(np.repeat(leng, self.num_active_dimensions))
            return lg

        k_object = self.pick_kernel_object(latent_kernel)

        k_s = k_object(
            variance=f64(var_s),
            lengthscales=get_lg(leng_s),
            active_dims=tf.range(self.num_active_dimensions),
            name="kernel_s",
        )
        k_st = k_object(
            variance=f64(var_st),
            lengthscales=get_lg(leng_st),
            active_dims=tf.range(self.num_active_dimensions),
            name="kernel_st",
        )
        k_t = k_object(
            variance=f64(var_t),
            lengthscales=get_lg(leng_t),
            active_dims=tf.range(self.num_active_dimensions),
            name="kernel_t",
        )

        self.kernel = (
            FilterKernel(2, [0, 0], active_dims=[self.num_active_dimensions]) * k_s
            + FilterKernel(2, [0, 1], active_dims=[self.num_active_dimensions]) * k_st
            + FilterKernel(2, [1, 1], active_dims=[self.num_active_dimensions]) * k_t
        )

        if add_prior:
            self.assign_prior(lengthscale_prior_parameters, variance_prior_parameters)

    @property
    def num_latent_gps(self):
        return 3

    @property
    def latent_kernels(self):
        return [self.kernel.kernels[i].kernels[1] for i in range(3)]

    def assign_prior(self, lengthscale_prior_parameters, variance_prior_parameters):
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        loc, std = variance_prior_parameters

        for k in self.latent_kernels:
            k.variance.prior = tfd.TruncatedNormal(
                loc=f64(loc), scale=f64(std), low=f64(1e-6), high=f64(30), name="kernel_var_prior_TruncatedNormal"
            )
            k.lengthscales.prior = tfd.Gamma(
                f64(np.repeat(a_lengthscale, self.num_active_dimensions)),
                f64(np.repeat(b_lengthscale, self.num_active_dimensions)),
                name="kernel_len_prior_Gamma",
            )

    def get_source_parameters_trainable(self):
        return self.kernel.kernels[0].kernels[1].lengthscales.trainable

    def set_source_parameters_trainable(self, source_trainable: bool):
        for p in range(self.output_dimension - 1):
            set_trainable(self.kernel.kernels[0].kernels[1], source_trainable)

    def get_target_parameters_trainable(self):
        return self.kernel.kernels[2].kernels[1].lengthscales.trainable

    def set_target_parameters_trainable(self, target_trainable: bool):
        self.output_dimension
        set_trainable(self.kernel.kernels[1].kernels[1], target_trainable)
        set_trainable(self.kernel.kernels[2].kernels[1], target_trainable)


if __name__ == "__main__":
    B = FilterKernel(2, [0, 0])
    print(B(np.array([[0.0]])))
    print(B(np.array([[0.0]]), np.array([[1.0]])))
    print(B(np.array([[1.0]]), np.array([[0.0]])))
    print(B(np.array([[1.0]])))
    print("###")
    B = FilterKernel(2, [1, 0])
    print(B(np.array([[0.0]])))
    print(B(np.array([[0.0]]), np.array([[1.0]])))
    print(B(np.array([[1.0]]), np.array([[0.0]])))
    print(B(np.array([[1.0]])))
    print("###")
    B = FilterKernel(2, [0, 1])
    print(B(np.array([[0.0]])))
    print(B(np.array([[0.0]]), np.array([[1.0]])))
    print(B(np.array([[1.0]]), np.array([[0.0]])))
    print(B(np.array([[1.0]])))
    print("###")
    B = FilterKernel(2, [1, 1])
    print(B(np.array([[0.0]])))
    print(B(np.array([[0.0]]), np.array([[1.0]])))
    print(B(np.array([[1.0]]), np.array([[0.0]])))
    print(B(np.array([[1.0]])))
    print("###")

    k = FlexibleTransferKernel(
        [1.0, 1.0, 1.0], [1, 1, 1], 2, 2, False, None, None, LatentKernel.MATERN52, False, None, "name_hello"
    )
    print(k(np.array([[0.5, 0.4, 1.0]]), np.array([[0.2, 0.2, 0.0]])))
    """
    print_summary(k)
    k.set_source_parameters_trainable(False)
    print_summary(k)
    k.set_source_parameters_trainable(True)
    print_summary(k)
    k.set_target_parameters_trainable(False)
    print_summary(k)
    k.set_source_parameters_trainable(False)
    k.set_target_parameters_trainable(True)
    print_summary(k)
    """
