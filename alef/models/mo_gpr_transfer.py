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

from gpflow.likelihoods import Gaussian, SwitchedLikelihood
from gpflow.logdensities import multivariate_normal
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import GPModel, InputData, MeanAndVariance, RegressionData
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor
from alef.kernels.multi_output_kernels.base_transfer_kernel import BaseTransferKernel

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)

f64 = gpflow.utilities.to_default_float


"""
Some pieces of the following class TransferGPR is adapted from GPflow
(https://github.com/GPflow/GPflow/blob/develop/gpflow/models/gpr.py
Copyright 2016-2020 The GPflow Contributors, licensed under the Apache License 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree).
"""


class TransferGPR(GPModel, InternalDataTrainingLossMixin):
    """
    Data should be (x, y)
    x: [N, D+1] array, the last column are integers of output dimension indices
    y: [N, 2] array, the second column
    """

    def __init__(
        self,
        source_data: RegressionData,
        data: RegressionData,
        kernel: BaseTransferKernel,
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
        self.source_data = data_input_to_tensor(source_data)
        self.data = data_input_to_tensor(data)

        self.__empty_target = np.shape(data[0])[0] == 0
        self.Ls = None

    def compute_source_cholesky(self):
        Xs, Ys = self.source_data
        noise = tf.linalg.diag(tf.reshape(self.likelihood._partition_and_stitch([Ys], "_conditional_variance"), [-1]))
        K = self.kernel(Xs) + noise
        return tf.linalg.cholesky(K)

    def set_source_cholesky(self, L):
        self.Ls = f64(L)

    def full_gram_noisy_cholesky(self, X_s, X_t, Ls):
        Kst = self.kernel(X_s, X_t)
        noise_t = tf.linalg.diag(
            tf.reshape(self.likelihood._partition_and_stitch([X_t[..., -2:]], "_conditional_variance"), [-1])
        )
        Kt = self.kernel(X_t) + noise_t

        corner_T = tf.linalg.triangular_solve(Ls, Kst, lower=True, adjoint=False)
        corner = tf.einsum("...ij->...ji", corner_T)
        Lt = tf.linalg.cholesky(Kt - tf.matmul(corner, corner_T))
        return tf.concat([tf.concat([Ls, tf.zeros_like(corner_T)], axis=-1), tf.concat([corner, Lt], axis=-1)], axis=-2)

    def maximum_log_likelihood_objective(self):
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self):
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        if self.Ls is None:
            print("compute source Cholesky")
            Ls = self.compute_source_cholesky()
        else:
            Ls = self.Ls

        Xs, Ys = self.source_data
        Xt, Yt = self.data

        if self.__empty_target:
            m = self.mean_function(Xs)

            log_prob = multivariate_normal(Ys[..., :1], m[..., :1], Ls)
            return tf.reduce_sum(log_prob)
        else:
            X = tf.concat([Xs, Xt], axis=0)
            Y = tf.concat([Ys, Yt], axis=0)

            L = self.full_gram_noisy_cholesky(Xs, Xt, Ls)
            m = self.mean_function(X)

            log_prob = multivariate_normal(Y[..., :1], m[..., :1], L)
            return tf.reduce_sum(log_prob)

    def predict_f(self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        Xs, Ys = self.source_data
        Xt, Yt = self.data
        X_data = tf.concat([Xs, Xt], axis=0)
        Y_data = tf.concat([Ys, Yt], axis=0)

        err = Y_data[..., :1] - self.mean_function(X_data)
        Ls = self.compute_source_cholesky() if self.Ls is None else self.Ls

        Lm = self.full_gram_noisy_cholesky(Xs, Xt, Ls)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X_data, Xnew)

        conditional = gpflow.conditionals.util.base_conditional_with_lm
        f_mean_zero, f_var = conditional(
            kmn, Lm, knn, err, full_cov=full_cov, white=False
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


if __name__ == "__main__":
    from gpflow.utilities import print_summary
    from alef.kernels.multi_output_kernels.coregionalization_transfer_kernel import CoregionalizationTransferKernel

    p = np.reshape([False, True] * 50, [-1, 1])
    X = np.hstack((np.random.standard_normal([100, 2]), p))
    Y = np.hstack((np.random.standard_normal([100, 1]), p))
    k = CoregionalizationTransferKernel(
        base_variance=1.0,
        base_lengthscale=1.0,
        input_dimension=2,
        output_dimension=2,
        add_prior=False,
        lengthscale_prior_parameters=(1, 9),
        variance_prior_parameters=(1, 0.3),
        active_on_single_dimension=False,
        active_dimension=None,
        name="test",
    )

    model = TransferGPR(
        source_data=(X[p[:, -1] == 0], Y[p[:, -1] == 0]), data=(X[p[:, -1] == 1], Y[p[:, -1] == 1]), kernel=k
    )
    print_summary(model)

    optimizer = gpflow.optimizers.Scipy()
    opt_res = optimizer.minimize(model.training_loss, model.trainable_variables)
    model.kernel.set_source_parameters_trainable(False)
    print(opt_res)
    print_summary(model)

    model.set_source_cholesky(model.compute_source_cholesky().numpy())
    opt_res = optimizer.minimize(model.training_loss, model.trainable_variables)
    print(opt_res)
    print_summary(model)
