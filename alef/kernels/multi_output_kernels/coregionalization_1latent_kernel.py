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
from gpflow.utilities import set_trainable, print_summary
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np

from typing import Tuple, Union, Sequence

from alef.kernels.multi_output_kernels.base_multioutput_flattened_kernel import BaseMultioutputFlattenedKernel
from alef.kernels.multi_output_kernels.latent_kernel_enum import LatentKernel

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)

f64 = gpflow.utilities.to_default_float


class Coregionalization1LKernel(BaseMultioutputFlattenedKernel):
    """
    X: [N, D+1], the last column is task index
    D: input_dimension
    """

    def __init__(
        self,
        base_variance: float,
        base_lengthscale: Union[float, Sequence[float]],
        W_rank: int,
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

        k = k_object(variance=f64(base_variance), lengthscales=lg, active_dims=tf.range(self.num_active_dimensions))

        set_trainable(k.variance, False)

        B = gpflow.kernels.Coregion(output_dim=P, rank=W_rank, active_dims=[self.num_active_dimensions])

        # B.kappa = gpflow.Parameter(B.kappa.numpy(), transform=bijectors.Sigmoid())
        # B.W = gpflow.Parameter(
        #    B.W.numpy(),
        #    transform=bijectors.Sigmoid( low= f64(-4.0), high= f64(4.0) )
        # )

        self.kernel = k * B

        if add_prior:
            a_lengthscale, b_lengthscale = lengthscale_prior_parameters
            loc, std = variance_prior_parameters

            k.variance.prior = tfd.TruncatedNormal(
                loc=f64(loc), scale=f64(std), low=f64(1e-6), high=f64(30), name="kernel_var_prior_TruncatedNormal"
            )
            k.lengthscales.prior = tfd.Gamma(
                f64(np.repeat(a_lengthscale, self.num_active_dimensions)),
                f64(np.repeat(b_lengthscale, self.num_active_dimensions)),
                name="kernel_len_prior_Gamma",
            )
            B.W.prior = tfd.Normal(
                f64(np.zeros(B.W.shape)), f64(1 / W_rank * np.ones(B.W.shape)), name="kernel_W_prior_Normal"
            )
            B.kappa.assign(np.array([1.001e-6] * self.output_dimension))
            set_trainable(B.kappa, False)
            # B.kappa.prior = tfd.TruncatedNormal(
            #    loc = f64(loc), scale = f64(std),
            #    low = f64(1e-6), high = f64(30),
            #    name='kernel_kappa_prior_TruncatedNormal')

    @property
    def num_latent_gps(self):
        return 1

    @property
    def latent_kernels(self):
        return tuple(
            self.kernel.kernels[0],
        )


if __name__ == "__main__":
    k = Coregionalization1LKernel(
        base_variance=1.0,
        base_lengthscale=1.0,
        W_rank=2,
        input_dimension=2,
        output_dimension=2,
        add_prior=False,
        lengthscale_prior_parameters=(1, 9),
        variance_prior_parameters=(1, 0.3),
        latent_kernel=LatentKernel.MATERN52,
        active_on_single_dimension=False,
        active_dimension=None,
        name="test",
    )
    print_summary(k)
