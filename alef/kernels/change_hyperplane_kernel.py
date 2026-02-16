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

from typing import Tuple
import gpflow
from gpflow.utilities import positive
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)

f64 = gpflow.utilities.to_default_float


class ChangeHyperplaneKernel(gpflow.kernels.Kernel):
    def __init__(
        self,
        kernel_1: gpflow.kernels.Kernel,
        kernel_2: gpflow.kernels.Kernel,
        input_dimension: int,
        base_hyperplane_mu: float,
        base_hyperplane_std: float,
        base_smoothing: float,
        smoothing_prior_parameters: Tuple[float, float],
        **kwargs,
    ):
        super().__init__()
        dimension = input_dimension
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.smoothing_param = gpflow.Parameter(base_smoothing, transform=positive())
        a_smoothing, b_smoothing = smoothing_prior_parameters
        self.smoothing_param.prior = tfd.Gamma(f64(a_smoothing), f64(b_smoothing))
        self.w = gpflow.Parameter(np.repeat(base_hyperplane_mu, dimension + 1))
        self.w.prior = tfd.Normal(
            np.repeat(base_hyperplane_mu, dimension + 1), np.repeat(base_hyperplane_std, dimension + 1)
        )

    def sigmoid(self, x, smoothing):
        return 1 / (1 + tf.math.exp(-1 * x * smoothing))

    def get_left_weight(self, X):
        w_0 = self.w[0]
        w_rest = tf.expand_dims(self.w[1:], axis=0)
        return self.sigmoid(w_0 + tf.linalg.matmul(X, tf.transpose(w_rest)), self.smoothing_param)

    def get_right_weight(self, X):
        w_0 = self.w[0]
        w_rest = tf.expand_dims(self.w[1:], axis=0)
        return 1 - self.sigmoid(w_0 + tf.linalg.matmul(X, tf.transpose(w_rest)), self.smoothing_param)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        weight1_X = self.get_left_weight(X)
        weight1_X2 = self.get_left_weight(X2)
        weight2_X = self.get_right_weight(X)
        weight2_X2 = self.get_right_weight(X2)
        K = tf.matmul(weight1_X, tf.transpose(weight1_X2)) * self.kernel_1.K(X, X2) + tf.matmul(
            weight2_X, tf.transpose(weight2_X2)
        ) * self.kernel_2.K(X, X2)
        return K

    def K_diag(self, X):
        return tf.math.multiply(
            tf.squeeze(tf.math.pow(self.get_left_weight(X), 2.0)), self.kernel_1.K_diag(X)
        ) + tf.math.multiply(tf.squeeze(tf.math.pow(self.get_right_weight(X), 2.0)), self.kernel_2.K_diag(X))


if __name__ == "__main__":
    ch_kernel = ChangeHyperplaneKernel(
        gpflow.kernels.RBF(1.0, [1.0, 1.0]), gpflow.kernels.RBF(1.0, [0.2, 0.2]), 2, 0.0, 1.0, 3.0, (2.0, 2.0)
    )
    X = np.array([[1.0, 2.0], [0.0, 0.0], [1.0, 1.0]])
    print(ch_kernel.K(X))
    print(ch_kernel.K_diag(X))
