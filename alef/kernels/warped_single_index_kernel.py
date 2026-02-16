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
from gpflow.utilities import positive
import tensorflow as tf
import numpy as np

from alef.kernels.warped_kernel_interface import WarpedKernelInterface

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)

f64 = gpflow.utilities.to_default_float


class WarpedSingleIndexKernel(gpflow.kernels.Kernel, WarpedKernelInterface):
    def __init__(
        self,
        base_alpha: float,
        base_beta: float,
        input_dimension: int,
        base_lengthscale: float,
        base_variance: float,
        **kwargs,
    ):
        super().__init__()
        dimension = input_dimension
        alphas = np.full(dimension, base_alpha)
        betas = np.full(dimension, base_beta)
        self.alphas = gpflow.Parameter(f64(alphas), transform=positive(), trainable=True)
        self.betas = gpflow.Parameter(f64(betas), transform=positive(), trainable=True)
        self.base_kernel = gpflow.kernels.RBF(lengthscales=f64(base_lengthscale), variance=f64(base_variance))

    def kumar_cdf(self, alphas, betas, x):
        x = tf.clip_by_value(x, 0.0, 1.0)
        out = 1 - tf.pow((1 - tf.pow(x, alphas)), betas)
        return out

    def warp(self, X: tf.Tensor) -> np.array:
        return self.kumar_cdf(self.alphas, self.betas, X)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return self.base_kernel.K(self.warp(X), self.warp(X2))

    def K_diag(self, X):
        return self.base_kernel.K_diag(self.warp(X))


if __name__ == "__main__":
    pass
