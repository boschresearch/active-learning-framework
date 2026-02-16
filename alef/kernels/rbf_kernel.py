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

from typing import Tuple, Union, Sequence
import numpy as np
import gpflow

from tensorflow_probability import distributions as tfd
from gpflow.utilities import set_trainable
from alef.kernels.base_elementary_kernel import BaseElementaryKernel
from alef.kernels.scale_interface import StationaryKernelGPflow

gpflow.config.set_default_float(np.float64)
f64 = gpflow.utilities.to_default_float


class RBFKernel(BaseElementaryKernel, StationaryKernelGPflow):
    has_fourier_feature = True

    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: Union[float, Sequence[float]],
        base_variance: float,
        fix_lengthscale: bool,
        fix_variance: bool,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name)
        if hasattr(base_lengthscale, "__len__"):
            if len(base_lengthscale) == 1:
                lg = f64(np.repeat(base_lengthscale[0], self.num_active_dimensions))
            else:
                assert len(base_lengthscale) == self.num_active_dimensions
                lg = f64(base_lengthscale)
        else:
            lg = f64(np.repeat(base_lengthscale, self.num_active_dimensions))

        self.kernel = gpflow.kernels.RBF(
            lengthscales=lg,
            variance=f64([base_variance]),
        )
        if fix_lengthscale:
            set_trainable(self.kernel.lengthscales, False)
        if fix_variance:
            set_trainable(self.kernel.variance, False)

        if add_prior:
            a_lengthscale, b_lengthscale = lengthscale_prior_parameters
            a_variance, b_variance = variance_prior_parameters
            self.kernel.lengthscales.prior = tfd.Gamma(
                f64(np.repeat(a_lengthscale, self.num_active_dimensions)),
                f64(np.repeat(b_lengthscale, self.num_active_dimensions)),
            )
            self.kernel.variance.prior = tfd.Gamma(f64([a_variance]), f64([b_variance]))
