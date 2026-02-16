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
from typing import Tuple, List
from gpflow.utilities import print_summary, set_trainable

from tensorflow_probability import distributions as tfd

from alef.kernels.multi_output_kernels.base_transfer_kernel import BaseTransferKernel
from alef.kernels.multi_output_kernels.latent_kernel_enum import LatentKernel

gpflow.config.set_default_float(np.float64)
f64 = gpflow.utilities.to_default_float

"""
The kernel in

Alonso Marco, Felix Berkenkamp, Philipp Hennig, Angela P. Schoellig, Andreas Krause, Stefan Schaal and Sebastian Trimpe,
ICRA 2017, Virtual vs. Real: Trading Off Simulations and Physical Experiments in Reinforcement Learning with Bayesian Optimization


The kernel is similar to this one:
Matthias Poloczek, Jialei Wang and Peter Frazier, NeurIPS 2017, Multi-Information Source Optimization

"""


class MIAdditiveKernel(BaseTransferKernel):
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
        assert len(variance_list) == P
        assert len(lengthscale_list) == P
        super().__init__(
            input_dimension,
            P,
            active_on_single_dimension,
            active_dimension,
            name,
            **kwargs,
        )

        var_source = variance_list[0]
        var_error = variance_list[1]

        leng_source = lengthscale_list[0]
        leng_error = lengthscale_list[1]

        if hasattr(leng_source, "__len__"):
            if len(leng_source) == 1:
                lg = f64(np.repeat(leng_source[0], self.num_active_dimensions))
            else:
                assert len(leng_source) == self.num_active_dimensions
                lg = f64(leng_source)
        else:
            lg = f64(np.repeat(leng_source, self.num_active_dimensions))

        k_object = self.pick_kernel_object(latent_kernel)

        k_source = k_object(
            variance=f64(var_source),
            lengthscales=lg,
            active_dims=tf.range(self.num_active_dimensions),
            name="kernel_source",
        )

        if hasattr(leng_error, "__len__"):
            if len(leng_error) == 1:
                lg = f64(np.repeat(leng_error[0], self.num_active_dimensions))
            else:
                assert len(leng_error) == self.num_active_dimensions
                lg = f64(leng_error)
        else:
            lg = f64(np.repeat(leng_error, self.num_active_dimensions))

        k_error = k_object(
            variance=f64(var_error),
            lengthscales=lg,
            active_dims=tf.range(self.num_active_dimensions),
            name="kernel_error",
        )

        # dt = Delta_t(f64(1.0), P, active_dims=[self.num_active_dimensions] ) # !!! has bug
        dt = gpflow.kernels.Linear(active_dims=[self.num_active_dimensions])
        set_trainable(dt.variance, False)

        self.kernel = k_source + dt * k_error

        if add_prior:
            self.assign_prior(lengthscale_prior_parameters, variance_prior_parameters)

    @property
    def num_latent_gps(self):
        return 1

    @property
    def latent_kernels(self):
        return [self.kernel.kernels[0], self.kernel.kernels[1].kernels[1]]

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
        return self.kernel.kernels[0].lengthscales.trainable

    def set_source_parameters_trainable(self, source_trainable: bool):
        for p in range(self.output_dimension - 1):
            set_trainable(self.kernel.kernels[p], source_trainable)

    def get_target_parameters_trainable(self):
        return self.kernel.kernels[1].kernels[1].lengthscales.trainable

    def set_target_parameters_trainable(self, target_trainable: bool):
        self.output_dimension
        set_trainable(self.kernel.kernels[1].kernels[1], target_trainable)


if __name__ == "__main__":
    k = MIAdditiveKernel([1.0, 1.0], [1, 1], 2, 2, False, None, None, LatentKernel.MATERN52, False, None, "name_hello")
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
