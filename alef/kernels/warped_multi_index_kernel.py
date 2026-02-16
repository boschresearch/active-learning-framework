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

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)

f64 = gpflow.utilities.to_default_float


class WarpedMultiIndexKernel(gpflow.kernels.Kernel):
    def __init__(
        self,
        base_A_scale: float,
        base_alpha: float,
        base_beta: float,
        base_rho: float,
        input_dimension: int,
        base_lengthscale: float,
        base_variance=1.0,
        matrix_learnable=True,
        rho_learnable=False,
        **kwargs,
    ):
        super().__init__()
        dimension = input_dimension
        A = np.identity(dimension) * base_A_scale
        alphas = np.full(dimension, base_alpha)
        betas = np.full(dimension, base_beta)
        rhos = np.full(dimension, base_rho)
        self.alphas = gpflow.Parameter(f64(alphas), transform=positive(), trainable=True)
        self.betas = gpflow.Parameter(f64(betas), transform=positive(), trainable=True)
        self.rhos = gpflow.Parameter(f64(rhos), transform=positive(), trainable=rho_learnable)
        self.base_kernel = gpflow.kernels.Matern52(lengthscales=f64(base_lengthscale), variance=f64(base_variance))
        self.A = gpflow.Parameter(f64(A), trainable=matrix_learnable)

    def kumar_cdf(self, alphas, betas, x):
        x = tf.clip_by_value(x, 0.0, 1.0)
        out = 1 - tf.pow((1 - tf.pow(x, alphas)), betas)
        return out

    def psi(self, alphas, betas, rhos, x):
        out = 1 / (1 + rhos) * (rhos * (x + 0.5) + self.kumar_cdf(alphas, betas, x + 0.5)) - 0.5
        return out

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        lin_transformed_X = tf.linalg.matmul(X - 0.5, self.A)
        nonlin_transformed_X = tf.map_fn(lambda x: self.psi(self.alphas, self.betas, self.rhos, x), lin_transformed_X)
        lin_transformed_X2 = tf.linalg.matmul(X2 - 0.5, self.A)
        nonlin_transformed_X2 = tf.map_fn(lambda x: self.psi(self.alphas, self.betas, self.rhos, x), lin_transformed_X2)
        output = self.base_kernel.K(nonlin_transformed_X, nonlin_transformed_X2)
        return output

    def K_diag(self, X):
        lin_transformed = tf.linalg.matmul(X - 0.5, self.A)
        nonlin_transformed = tf.map_fn(lambda x: self.psi(self.alphas, self.betas, self.rhos, x), lin_transformed)
        output = self.base_kernel.K_diag(nonlin_transformed)
        return output


if __name__ == "__main__":
    kernel = WarpedMultiIndexKernel(
        A=np.array([[1, 0], [0, 1]]), alphas=np.array([1.0, 1.0]), betas=np.array([1.0, 1.0]), rhos=np.array([1.0, 1.0])
    )
    # print(kernel.phi(np.array([1.0,1.0]),np.array([1.0,0.5]),np.array([0.5,0.6])))
    print(kernel.K(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])))
