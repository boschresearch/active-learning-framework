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
from typing import Tuple
import numpy as np

from tensorflow_probability import distributions as tfd
from alef.kernels.base_object_kernel import BaseObjectKernel

gpflow.config.set_default_float(np.float64)
f64 = gpflow.utilities.to_default_float


class RBFObjectKernel(BaseObjectKernel):
    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: float,
        base_variance: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        **kwargs,
    ):
        super().__init__()
        self.kernel = gpflow.kernels.RBF(
            lengthscales=f64(np.repeat(base_lengthscale, input_dimension)), variance=f64([base_variance])
        )
        if add_prior:
            a_lengthscale, b_lengthscale = lengthscale_prior_parameters
            a_variance, b_variance = variance_prior_parameters
            self.kernel.lengthscales.prior = tfd.Gamma(
                f64(np.repeat(a_lengthscale, input_dimension)), f64(np.repeat(b_lengthscale, input_dimension))
            )
            self.kernel.variance.prior = tfd.Gamma(f64([a_variance]), f64([b_variance]))

    def K(self, X, X2=None):
        assert isinstance(X, list)
        X = np.array(X)
        if X2 is None:
            X2 = X
        else:
            X2 = np.array(X2)
        return self.kernel.K(X, X2)

    def K_diag(self, X):
        X = np.array(X)
        return self.kernel.K_diag(X)
