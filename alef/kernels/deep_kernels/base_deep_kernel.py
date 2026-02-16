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
import numpy as np
from tensorflow_probability import distributions as tfd
import tensorflow as tf
from alef.kernels.regularized_kernel_interface import RegularizedKernelInterface
from alef.models.feature_extractors.base_feature_extractor import BaseFeatureExtractor

gpflow.config.set_default_float(np.float64)
f64 = gpflow.utilities.to_default_float


class BaseDeepKernel(gpflow.kernels.Kernel, RegularizedKernelInterface):
    def __init__(
        self,
        input_dimension: int,
        feature_extractor: BaseFeatureExtractor,
        base_lengthscale: float,
        base_variance: float,
        lengthscale_trainable: bool,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        **kwargs,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_extractor.set_mode(False)
        self.input_dimension = input_dimension
        assert self.input_dimension == self.feature_extractor.get_input_dimension()
        self.feature_dimension = self.feature_extractor.get_output_dimension()
        self.base_kernel = gpflow.kernels.Matern52(
            lengthscales=f64(np.repeat(base_lengthscale, self.feature_dimension)), variance=f64([base_variance])
        )
        gpflow.set_trainable(self.base_kernel.lengthscales, lengthscale_trainable)
        if add_prior:
            a_lengthscale, b_lengthscale = lengthscale_prior_parameters
            a_variance, b_variance = variance_prior_parameters
            self.base_kernel.lengthscales.prior = tfd.Gamma(
                f64(np.repeat(a_lengthscale, self.feature_dimension)),
                f64(np.repeat(b_lengthscale, self.feature_dimension)),
            )
            self.base_kernel.variance.prior = tfd.Gamma(f64([a_variance]), f64([b_variance]))

    def K(self, X, X2=None):
        if X2 is None:
            features = self.feature_extractor.forward(X)
            return self.base_kernel.K(features, features)
        return self.base_kernel.K(self.feature_extractor.forward(X), self.feature_extractor.forward(X2))

    def K_diag(self, X):
        return self.base_kernel.K_diag(self.feature_extractor.forward(X))

    def regularization_loss(self, x_data: np.array) -> tf.Tensor:
        loss = self.feature_extractor.regularization_loss(x_data)
        return loss

    def set_mode(self, training_mode: bool):
        self.feature_extractor.set_mode(training_mode)
