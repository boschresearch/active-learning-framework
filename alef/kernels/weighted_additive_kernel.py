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

from typing import List, Tuple
from tensorflow_probability import distributions as tfd
import numpy as np
import gpflow
import tensorflow as tf
from alef.kernels.base_elementary_kernel import BaseElementaryKernel
from alef.utils.utils import sigmoid_tf
from gpflow.utilities import positive

f64 = gpflow.utilities.to_default_float


class WeightedAdditiveKernel(gpflow.kernels.Kernel):
    def __init__(
        self,
        base_kernel_list: List[gpflow.kernels.Kernel],
        base_variance: float,
        use_own_variance: bool,
        add_prior: bool,
        alpha_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        **kwargs,
    ):
        super().__init__()
        self.base_kernel_list = base_kernel_list
        self.use_own_variance = use_own_variance
        if self.use_own_variance:
            for kernel in self.base_kernel_list:
                assert isinstance(kernel, BaseElementaryKernel)
                if hasattr(kernel.kernel, "variance"):
                    gpflow.set_trainable(kernel.kernel.variance, False)
                    # kernel.kernel.variance.trainable = False
        self.variance = gpflow.Parameter(f64([base_variance]), trainable=self.use_own_variance, transform=positive())
        n_kernels = len(base_kernel_list)
        self.alphas = gpflow.Parameter(f64(np.repeat(0.0, n_kernels)))
        alpha_mu, alpha_sigma = alpha_prior_parameters
        variance_a, variance_b = variance_prior_parameters
        if add_prior:
            self.alphas.prior = tfd.Normal(np.repeat(alpha_mu, n_kernels), np.repeat(alpha_sigma, n_kernels))
            self.variance.prior = tfd.Gamma(f64([variance_a]), f64([variance_b]))

    def get_weights(self):
        sigmoid_alphas = sigmoid_tf(self.alphas)
        weights = sigmoid_alphas / tf.reduce_sum(sigmoid_alphas)
        return weights

    def K(self, X, X2=None):
        weights = self.get_weights()
        K = tf.add_n([weights[i] * k.K(X, X2) for i, k in enumerate(self.base_kernel_list)])
        if self.use_own_variance:
            return self.variance * K
        else:
            return K

    def K_diag(self, X):
        weights = self.get_weights()
        K_diag = tf.add_n([weights[i] * k.K_diag(X) for i, k in enumerate(self.base_kernel_list)])
        if self.use_own_variance:
            return self.variance * K_diag
        else:
            K_diag


if __name__ == "__main__":
    kernel1 = gpflow.kernels.RBF(variance=2.0)
    kernel2 = gpflow.kernels.RBF(lengthscales=0.5)
    x = np.array([[1.0], [2.0], [3.0]])
    x2 = np.array([[1.0], [2.2], [2.5]])
    print(kernel1.K(x, x2))
    print(kernel2.K(x, x2))
    kernel = WeightedAdditiveKernel([kernel1, kernel2], False, (0.0, 1.0))
    print(kernel.K(x, x2))
