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

from typing import Optional, Union, Sequence
import numpy as np
import tensorflow as tf
import gpflow

from gpflow.kernels import Kernel
from gpflow.likelihoods import Gaussian, SwitchedLikelihood
from gpflow.logdensities import multivariate_normal
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import GPModel, InputData, MeanAndVariance, RegressionData
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)

f64 = gpflow.utilities.to_default_float

"""
Some pieces of the following class SOMOGPR is adapted from GPflow
(https://github.com/GPflow/GPflow/blob/develop/gpflow/models/gpr.py
Copyright 2016-2020 The GPflow Contributors, licensed under the Apache License 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree).
"""


class SOMOGPR(GPModel, InternalDataTrainingLossMixin):
    """
    Data should be (x, y)
    x: [N, D+1] array, the last column are integers of output dimension indices
    y: [N, 2] array, the second column

    This model uses SwitchedLikelihood.
    If one observation noise for all output dims is enough for you, then just use gpflow.models.GPR
    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: Union[float, Sequence[float]] = 1.0,
    ):
        num_latent_gps = kernel.num_latent_gps
        output_dimension = kernel.output_dimension
        if hasattr(noise_variance, "__len__"):
            assert len(noise_variance) == output_dimension
            noise_variance = f64(np.array(noise_variance))
        else:
            noise_variance = f64(np.repeat(noise_variance, output_dimension))

        lik_list = [Gaussian(var) for var in noise_variance]
        likelihood = SwitchedLikelihood(lik_list)

        super().__init__(kernel, likelihood, mean_function, num_latent_gps=num_latent_gps)
        self.data = data_input_to_tensor(data)

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def _lml(self, X, Y):
        K = self.kernel(X)  # [N, N]
        s_diag = tf.reshape(self.likelihood._partition_and_stitch([Y], "_conditional_variance"), [-1])

        L = tf.linalg.cholesky(K + tf.linalg.diag(s_diag))
        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y[..., :1], m[..., :1], L)
        return tf.reduce_sum(log_prob)

    def conditional_log_marginal_likelihood(self):
        X, Y = self.data  # [N, D+1], [N, 2]
        Xp, Yp = self.pdata

        return self._lml(tf.concat([X, Xp], axis=0), tf.concat([Y, Yp], axis=0)) - self._lml(X, Y)

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        X, Y = self.data  # [N, D+1], [N, 2]

        return self._lml(X, Y)

    def predict_f(self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        X_data, Y_data = self.data
        err = Y_data[..., :1] - self.mean_function(X_data)

        kmm = self.kernel(X_data)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X_data, Xnew)

        s = tf.linalg.diag(tf.reshape(self.likelihood._partition_and_stitch([Y_data], "_conditional_variance"), [-1]))

        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm + s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var

    def predict_y(self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False) -> MeanAndVariance:
        f_mean, f_var = self.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)

        s = self.likelihood._partition_and_stitch([Xnew[..., -2:]], "_conditional_variance")

        if full_cov:
            return f_mean, f_var + tf.linalg.diag(tf.reshape(s, [-1]))
        else:
            return f_mean, f_var + s
