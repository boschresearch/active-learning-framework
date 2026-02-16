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
from gpflow.utilities import set_trainable, print_summary, positive
from gpflow.base import Parameter
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np

from typing import Tuple, List

from alef.kernels.multi_output_kernels.base_multioutput_kernel import BaseMultioutputKernel
from alef.kernels.multi_output_kernels.base_multioutput_flattened_kernel import BaseMultioutputFlattenedKernel
from alef.kernels.multi_output_kernels.latent_kernel_enum import LatentKernel

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)

f64 = gpflow.utilities.to_default_float

"""
Some pieces of the following class FlattenedLinearCoregionalization is adapted from GPflow
(https://github.com/GPflow/GPflow/blob/develop/gpflow/kernels/multioutput/kernels.py
Copyright 2016-2020 The GPflow Contributors, licensed under the Apache License 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree).
"""


class SeparatedWLinearCoregionalization(gpflow.kernels.LinearCoregionalization):
    def __init__(self, kernels, W_list, name=None):
        assert len(kernels) == len(W_list)
        self.W_list = [Parameter(W_l) for W_l in W_list]
        # self.W_list = [Parameter(W_l, transform=bijectors.Sigmoid( low= f64(-4.0), high= f64(4.0) ) ) for W_l in W_list]
        super().__init__(kernels, tf.concat([tf.reshape(W_l, [-1, 1]) for W_l in W_list], axis=1), name)
        delattr(self, "W")

    @property
    def num_latent_gps(self):
        return len(self.W_list)  # L

    def K(self, *args, **kwargs):
        # self.W = tf.einsum("...ij->...ji", W_transpose)
        self.W = tf.concat([tf.reshape(W_l, [-1, 1]) for W_l in self.W_list], axis=1)
        output = super().K(*args, **kwargs)
        delattr(self, "W")
        return output

    def K_diag(self, *args, **kwargs):
        self.W = tf.concat([tf.reshape(W_l, [-1, 1]) for W_l in self.W_list], axis=1)
        output = super().K_diag(*args, **kwargs)
        delattr(self, "W")
        return output


class FlattenedLinearCoregionalization(SeparatedWLinearCoregionalization):
    """
    X: [N, D+1], the last column is task index
    D: input_dimension
    """

    def __init__(
        self, kernels, W_list, input_dimension, add_error_kernel: bool = False, error_variance=None, name=None
    ):
        super().__init__(kernels, W_list, name)
        self.input_dimension = input_dimension
        if add_error_kernel:
            self.error_variance = Parameter(error_variance, transform=positive())

    def K(self, X, X2=None, full_output_cov: bool = False):
        """
        in this method full_output_cov is useless,
        but LinearCoregionalization, which inherit gpflow.kernels.MultioutputKernel, needs this argument
        """
        D = self.input_dimension

        p = tf.cast(tf.gather(X, D, axis=-1), tf.int32)
        x = tf.gather(X, tf.range(D), axis=-1)

        p2 = p if X2 is None else tf.cast(tf.gather(X2, D, axis=-1), tf.int32)
        x2 = x if X2 is None else tf.gather(X2, tf.range(D), axis=-1)

        W = tf.concat([tf.reshape(W_l, [-1, 1]) for W_l in self.W_list], axis=1)

        Kxx = self.Kgg(x, x2)  # [L, N, N2]
        wX = tf.transpose(tf.gather(W, p))  # [L, N]
        wX2 = tf.transpose(tf.gather(W, p2))  # [L, N2]

        if hasattr(self, "error_variance"):
            p = tf.cast(p, x.dtype)
            p = tf.expand_dims(p, axis=-1)
            p2 = tf.cast(p2, x2.dtype)
            p2 = tf.expand_dims(p2, axis=-1)
            return tf.einsum("...lm,...ln,...lmn->...mn", wX, wX2, Kxx) + tf.tensordot(
                p * self.error_variance, p2, [[-1], [-1]]
            )
        else:
            # return tf.reduce_sum(wX[:, :, None] * Kxx * wX2[:, None,:], axis=0) # [N, N2]
            return tf.einsum("...lm,...ln,...lmn->...mn", wX, wX2, Kxx)

    def K_diag(self, X, full_output_cov: bool = False):
        """
        in this method full_output_cov is useless,
        but LinearCoregionalization, which inherit gpflow.kernels.MultioutputKernel, needs this argument
        """
        D = self.input_dimension

        p = tf.cast(tf.gather(X, D, axis=-1), tf.int32)
        x = tf.gather(X, tf.range(D), axis=-1)

        K = tf.stack([k.K_diag(x) for k in self.kernels], axis=1)  # [N, L]
        W = tf.concat([tf.reshape(W_l, [-1, 1]) for W_l in self.W_list], axis=1)
        w2X = tf.gather(W**2, p)  # [N, L]
        if hasattr(self, "error_variance"):
            p = tf.cast(p, x.dtype)
            p = tf.expand_dims(p, axis=-1)
            return tf.einsum("...nl,...nl->...n", w2X, K) + tf.reduce_sum(tf.square(p) * self.error_variance, axis=-1)
        else:
            return tf.einsum("...nl,...nl->...n", w2X, K)


class CoregionalizationSOKernel(BaseMultioutputFlattenedKernel):
    """
    X: [N, D+1], the last column is task index
    D: input_dimension
    """

    def __init__(
        self,
        variance_list: List,
        lengthscale_list: List,
        input_dimension: int,
        output_dimension: int,
        add_error_kernel: bool,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        latent_kernel: LatentKernel,
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        if len(variance_list) != len(lengthscale_list):
            raise ValueError("need to have same number of variances and lengthscales")

        super().__init__(
            input_dimension,
            output_dimension,
            active_on_single_dimension,
            active_dimension,
            name,
            **kwargs,
        )

        P = output_dimension
        L = len(variance_list)
        k_object = self.pick_kernel_object(latent_kernel)

        kernel_list = []
        for sig2, l in zip(variance_list, lengthscale_list):
            if hasattr(l, "__len__"):
                if len(l) == 1:
                    lg = f64(np.repeat(l[0], self.num_active_dimensions))
                else:
                    assert len(l) == self.num_active_dimensions
                    lg = f64(l)
            else:
                lg = f64(np.repeat(l, self.num_active_dimensions))

            k = k_object(variance=f64(sig2), lengthscales=lg)
            set_trainable(k.variance, False)

            kernel_list.append(k)

        # W = np.random.normal(size=[P, L])
        W_list = [f64(np.random.normal(size=[P])) for _ in range(L)]

        self.kernel = FlattenedLinearCoregionalization(
            kernel_list,
            W_list=W_list,
            input_dimension=self.num_active_dimensions,
            add_error_kernel=add_error_kernel,
            error_variance=0.1,
        )

        if add_prior:
            self.assign_prior(lengthscale_prior_parameters, variance_prior_parameters)

    @property
    def num_latent_gps(self):
        return self.kernel.num_latent_gps

    @property
    def latent_kernels(self):
        return self.kernel.latent_kernels

    def assign_prior(self, lengthscale_prior_parameters, variance_prior_parameters):
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        loc, std = variance_prior_parameters

        if hasattr(self.kernel, "error_variance"):
            self.kernel.error_variance.prior = tfd.TruncatedNormal(
                loc=f64(0.1),
                scale=f64(np.sqrt(0.1) * std),
                low=f64(1e-6),
                high=f64(30),
                name="kernel_var_prior_TruncatedNormal",
            )

        for k in self.kernel.kernels:
            k.variance.prior = tfd.TruncatedNormal(
                loc=f64(loc), scale=f64(std), low=f64(1e-6), high=f64(30), name="kernel_var_prior_TruncatedNormal"
            )
            k.lengthscales.prior = tfd.Gamma(
                f64(np.repeat(a_lengthscale, self.num_active_dimensions)),
                f64(np.repeat(b_lengthscale, self.num_active_dimensions)),
                name="kernel_len_prior_Gamma",
            )

        P = self.output_dimension
        L = self.num_latent_gps
        for W_l in self.kernel.W_list:
            W_l.prior = tfd.Normal(loc=f64(np.zeros(P)), scale=f64(1 / L * np.ones(P)), name="kernel_W_prior_Normal")


class CoregionalizationMOKernel(BaseMultioutputKernel):
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
        if len(variance_list) != len(lengthscale_list):
            raise ValueError("need to have same number of variances and lengthscales")
        super().__init__(
            input_dimension,
            output_dimension,
            active_on_single_dimension,
            active_dimension,
            name,
            **kwargs,
        )

        P = output_dimension
        L = len(variance_list)
        k_object = self.pick_kernel_object(latent_kernel)

        kernel_list = []
        for sig2, l in zip(variance_list, lengthscale_list):
            if hasattr(l, "__len__"):
                if len(l) == 1:
                    lg = f64(np.repeat(l[0], self.num_active_dimensions))
                else:
                    assert len(l) == self.num_active_dimensions
                    lg = f64(l)
            else:
                lg = f64(np.repeat(l, self.num_active_dimensions))

            k = k_object(variance=f64(sig2), lengthscales=lg)
            set_trainable(k.variance, False)

            kernel_list.append(k)

        # W = np.random.normal(size=[P, L])
        # self.kernel = gpflow.kernels.LinearCoregionalization(kernel_list, W=f64(W))

        # W_T = np.random.normal(size=[L, P])
        W_list = [f64(np.random.normal(size=[P])) for _ in range(L)]

        self.kernel = SeparatedWLinearCoregionalization(kernel_list, W_list=W_list)
        if add_prior:
            self.assign_prior(lengthscale_prior_parameters, variance_prior_parameters)

    def assign_prior(self, lengthscale_prior_parameters, variance_prior_parameters):
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        loc, std = variance_prior_parameters

        for k in self.kernel.kernels:
            k.variance.prior = tfd.TruncatedNormal(
                loc=f64(loc), scale=f64(std), low=f64(1e-6), high=f64(30), name="kernel_var_prior_TruncatedNormal"
            )
            k.lengthscales.prior = tfd.Gamma(
                f64(np.repeat(a_lengthscale, self.num_active_dimensions)),
                f64(np.repeat(b_lengthscale, self.num_active_dimensions)),
                name="kernel_len_prior_Gamma",
            )

        P = self.output_dimension
        L = self.num_latent_gps
        for W_l in self.kernel.W_list:
            W_l.prior = tfd.Normal(
                loc=f64(np.zeros([P])), scale=f64(1 / L * np.ones([P])), name="kernel_W_prior_Normal"
            )


if __name__ == "__main__":
    k = CoregionalizationSOKernel(
        [1.0, 1.0],
        [1.0, 1.0],
        2,
        3,
        add_error_kernel=True,
        add_prior=True,
        lengthscale_prior_parameters=(1, 9),
        variance_prior_parameters=(1, 0.3),
        latent_kernel=LatentKernel.MATERN52,
        active_on_single_dimension=False,
        active_dimension=None,
        name="hello",
    )
    print_summary(k)
    k = CoregionalizationMOKernel(
        [1.0, 1.0],
        [1.0, 1.0],
        2,
        3,
        add_error_kernel=True,
        add_prior=True,
        lengthscale_prior_parameters=(1, 9),
        variance_prior_parameters=(1, 0.3),
        latent_kernel=LatentKernel.MATERN52,
        active_on_single_dimension=False,
        active_dimension=None,
        name="hello",
    )
    print_summary(k)
