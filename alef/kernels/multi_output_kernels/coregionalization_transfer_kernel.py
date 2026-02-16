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
from gpflow.utilities import set_trainable, print_summary, Parameter, positive
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np

from typing import Tuple, Union, Sequence, Dict

from alef.kernels.multi_output_kernels.base_transfer_kernel import BaseTransferKernel
from alef.kernels.multi_output_kernels.latent_kernel_enum import LatentKernel
from alef.utils.gpflow_addon.kronecker_delta_kernel import Delta_t

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)

f64 = gpflow.utilities.to_default_float


class CoregionTransfer(gpflow.kernels.Kernel):
    def __init__(
        self,
        output_dim: int,
        rank: int,
        *,
        active_dims=None,
        name=None,
    ):
        super().__init__(active_dims=active_dims, name=name)
        self.output_dim = output_dim
        self.rank = rank

        self.W_s = Parameter(0.1 * np.ones((self.output_dim - 1, self.rank)))
        self.kappa_s = Parameter(1.0 * np.ones(self.output_dim - 1), transform=positive())
        self.W_t = Parameter(0.1 * np.ones([1, self.rank]))
        self.kappa_t = Parameter(1.0 * np.ones(1), transform=positive())

    @property
    def B(self):
        Bs = tf.linalg.matmul(self.W_s, self.W_s, transpose_b=True) + tf.linalg.diag(self.kappa_s)
        Bst = tf.linalg.matmul(self.W_s, self.W_t, transpose_b=True)
        Bts = tf.einsum("...ij-> ...ji", Bst)
        Bt = tf.linalg.matmul(self.W_t, self.W_t, transpose_b=True) + tf.linalg.diag(self.kappa_t)

        return tf.concat([tf.concat([Bs, Bst], axis=-1), tf.concat([Bts, Bt], axis=-1)], axis=-2)

    @property
    def B_diag(self):
        Bs_diag = tf.reduce_sum(tf.square(self.W_s), 1) + self.kappa_s
        Bt_diag = tf.reduce_sum(tf.square(self.W_t), 1) + self.kappa_t
        return tf.concat([Bs_diag, Bt_diag], axis=-1)

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


class CoregionalizationTransferKernel(BaseTransferKernel):
    """
    X: [N, D+1], the last column is task index
    D: input_dimension
    """

    def __init__(
        self,
        base_variance: float,
        base_lengthscale: Union[float, Sequence[float]],
        input_dimension: int,
        output_dimension: int,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        latent_kernel: LatentKernel,
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        *,
        fix_kernel: bool = False,
        assign_values: bool = False,
        parameter_values: Dict = {},
        **kwargs,
    ):
        super().__init__(
            input_dimension,
            output_dimension,
            active_on_single_dimension,
            active_dimension,
            name,
            **kwargs,
        )
        if hasattr(base_lengthscale, "__len__"):
            if len(base_lengthscale) == 1:
                lg = f64(np.repeat(base_lengthscale[0], self.num_active_dimensions))
            else:
                assert len(base_lengthscale) == self.num_active_dimensions
                lg = f64(base_lengthscale)
        else:
            lg = f64(np.repeat(base_lengthscale, self.num_active_dimensions))

        P = output_dimension
        k_object = self.pick_kernel_object(latent_kernel)

        ks = k_object(variance=f64(base_variance), lengthscales=lg, active_dims=tf.range(self.num_active_dimensions))
        B = CoregionTransfer(P, 1, active_dims=[self.num_active_dimensions])

        k = gpflow.utilities.deepcopy(ks)
        set_trainable(ks.variance, False)

        dt = Delta_t(f64(1.0), P, active_dims=[self.num_active_dimensions])
        set_trainable(dt.variance, False)

        one_kernel = ks * B + k * dt

        if add_prior:
            a_lengthscale, b_lengthscale = lengthscale_prior_parameters
            loc, std = variance_prior_parameters

            ks.variance.prior = tfd.TruncatedNormal(
                loc=f64(loc), scale=f64(std), low=f64(1e-6), high=f64(30), name="kernel_var_prior_TruncatedNormal"
            )
            ks.lengthscales.prior = tfd.Gamma(
                f64(np.repeat(a_lengthscale, self.num_active_dimensions)),
                f64(np.repeat(b_lengthscale, self.num_active_dimensions)),
                name="kernel_len_prior_Gamma",
            )
            k.lengthscales.prior = tfd.Gamma(
                f64(np.repeat(a_lengthscale, self.num_active_dimensions)),
                f64(np.repeat(b_lengthscale, self.num_active_dimensions)),
                name="kernel_len_prior_Gamma",
            )

        kernel_sum = one_kernel
        for p in range(1, P):
            kernel_sum = kernel_sum + gpflow.utilities.deepcopy(one_kernel)

        self.kernel = kernel_sum

        if assign_values:
            self.assign_parameters(parameter_values)
        if fix_kernel:
            set_trainable(self.kernel, False)

    @property
    def num_latent_gps(self):
        return self.output_dimension * 2

    @property
    def latent_kernels(self):
        lk_list = []
        for k in self.kernel.kernels:
            lk_list.append(k.kernels[0])
        return tuple(lk_list)

    def get_source_parameters_trainable(self):
        return self.kernel.kernels[0].kernels[0].lengthscales.trainable

    def set_source_parameters_trainable(self, source_trainable: bool):
        for p in range(self.output_dimension):
            set_trainable(self.kernel.kernels[2 * p].kernels[0].lengthscales, source_trainable)
            set_trainable(self.kernel.kernels[2 * p].kernels[1].W_s, source_trainable)
            set_trainable(self.kernel.kernels[2 * p].kernels[1].kappa_s, source_trainable)

    def get_target_parameters_trainable(self):
        return self.kernel.kernels[1].kernels[0].lengthscales.trainable

    def set_target_parameters_trainable(self, target_trainable: bool):
        for p in range(self.output_dimension):
            set_trainable(self.kernel.kernels[2 * p].kernels[1].W_t, target_trainable)
            set_trainable(self.kernel.kernels[2 * p].kernels[1].kappa_t, target_trainable)
            set_trainable(self.kernel.kernels[2 * p + 1].kernels[0].variance, target_trainable)
            set_trainable(self.kernel.kernels[2 * p + 1].kernels[0].lengthscales, target_trainable)


if __name__ == "__main__":
    k = CoregionalizationTransferKernel(
        base_variance=1.0,
        base_lengthscale=1.0,
        input_dimension=2,
        output_dimension=2,
        add_prior=True,
        lengthscale_prior_parameters=(1, 9),
        variance_prior_parameters=(1, 0.3),
        latent_kernel=LatentKernel.MATERN52,
        active_on_single_dimension=False,
        active_dimension=None,
        name="test",
    )
    print_summary(k)

    X = np.hstack((np.random.standard_normal(size=[10, 2]), np.reshape([False, True] * 5, [10, 1])))

    print(k.trainable_variables)
    # print(k.kernel.kernels[0].kernels[1].B)
    # print(k.kernel.kernels[0].kernels[1].B_diag)
    # print(k(X))
    # print(k(X, X))
