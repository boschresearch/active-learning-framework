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

import numpy as np
import tensorflow as tf

from alef.utils.gpflow_addon.multi_variance_likelihood import MultiGaussian

from gpflow.logdensities import multivariate_normal
from gpflow.models.model import GPModel, InputData, MeanAndVariance
from gpflow.kernels import MultioutputKernel
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor

"""
Some pieces of the following class MOGPR is adapted from GPflow
(https://github.com/GPflow/GPflow/blob/develop/gpflow/models/gpr.py
Copyright 2016-2020 The GPflow Contributors, licensed under the Apache License 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree).
"""


class MOGPR(GPModel, InternalDataTrainingLossMixin):
    def __init__(self, data, kernel, noise_variance=1.0, mean_function=None, num_latent_gps: int = None):
        _, Y_data = data
        self.num_data, self.num_task = np.shape(Y_data)
        if hasattr(noise_variance, "__len__"):
            lik = MultiGaussian(np.array(noise_variance))
        else:
            lik = MultiGaussian(np.array([noise_variance for _ in range(np.shape(Y_data)[1])]))

        if num_latent_gps is None or isinstance(kernel, MultioutputKernel):
            """
            for MultioutputKernel,
            GPModel.calc_num_latent_gps_from_data(...)
            returns kernel.num_latent_gps
            """
            num_latent_gps = self.calc_num_latent_gps_from_data(data, kernel, lik)

        self.data = data_input_to_tensor(data)
        super().__init__(kernel, lik, mean_function, num_latent_gps)

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def _add_noise_cov(self, K: tf.Tensor) -> tf.Tensor:
        """
        Returns K + σ² I, where σ² is the likelihood noise variance,
        and I is the corresponding identity matrix.

        K has shape equal to either [N, N] or [N, P, N, P]
        """

        var = self.likelihood.variance
        N = tf.shape(K)[0]  # notice that self.num_data may not == N due to _aggregate_data()
        P = self.num_task

        k_reshape = tf.reshape(K, [N * P, N * P])

        k_diag = tf.linalg.diag_part(k_reshape)
        s_diag = tf.tile(var, [N])
        Ks = tf.linalg.set_diag(k_reshape, k_diag + s_diag)

        return tf.reshape(Ks, tf.shape(K))

    def _squeeze_kernel(self, K, squeeze_part: str = "full"):
        r"""
        input: K: kernel, shape [N1, P, N2, P]
               squeeze_part: 'full' or 'right'

        output: kernel, shape [PN1, PN2] or [N1, P, PN2]

        where output[i::P, j::P] = K[:, i, :, j]
        or    output[:,:,j::P] = K[..., j]
        """

        N1, P, N2, _ = tf.shape(K)  # notice that self.num_data may not == N due to _aggregate_data()

        if squeeze_part.lower() == "full":
            return tf.reshape(K, [N1 * P, N2 * P])
        elif squeeze_part.lower() == "right":
            return tf.reshape(K, [N1, P, N2 * P])
        else:
            raise ValueError("unknown input")

    def _aggregate_data(self):
        X, Y = self.data
        X = np.array(X)
        Y = np.array(Y)

        _, D = np.shape(X)
        P = self.num_task

        dtype = X.dtype.descr * D
        struct = X.view(dtype)

        _, idx, count = np.unique(struct, return_index=True, return_counts=True)

        X_aggr = tf.constant(X[idx].reshape([-1, D]))
        Y_aggr = Y[idx].reshape([-1, P])

        for j, c in enumerate(count):
            if c == 1:
                continue
            Yj = Y[np.in1d(struct, struct[idx[j]])]

            for p in range(P):
                entry = np.isfinite(Yj[:, p])
                if entry.sum() == 1:
                    Y_aggr[j, p] = Yj[entry, p]
                elif entry.sum() > 1:
                    ele = np.unique(Yj[entry, p])
                    if len(ele) > 1:
                        raise ValueError(
                            "ambiguous output, make sure each input only has 1 value for each element of the corresponding output"
                        )
                    Y_aggr[j, p] = ele

        Y_aggr = tf.constant(Y_aggr)

        return (X_aggr, Y_aggr)

    def _return_KNN_with_observed_outputs(self):
        X, Y = self._aggregate_data()

        # kernel with likelihood variance(s)
        K = self.kernel(
            X, full_cov=True, full_output_cov=True
        )  # full_output_cov is default to True anyway, just to avoid confusion

        N = tf.shape(K)[0]  # notice that self.num_data may not == N due to _aggregate_data()
        P = self.num_task

        # if noise_free:
        #    ks = K
        # else:
        ks = self._add_noise_cov(K)

        # re-shaping & dealing with unobserved outputs
        ks = tf.reshape(ks, [N * P, N * P])
        Y = tf.reshape(Y, [-1, 1])
        """
        notice: if P==1
        observed_entry == all true
        so ks_observed == ks
        """
        observed_entry = tf.squeeze(tf.math.is_finite(Y))
        ks_observed = tf.boolean_mask(tf.boolean_mask(ks, observed_entry, axis=0), observed_entry, axis=1)
        Y_observed = tf.boolean_mask(Y, observed_entry, axis=0)

        return ks_observed, Y_observed, observed_entry

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta) = \log N(Y | m, K)
            = constant - 1/2 \log det(K) - 1/2 (Y-m)^T K^-1 (Y-m).

        """
        X, Y = self._aggregate_data()

        ks_observed, Y_observed, observed_entry = self._return_KNN_with_observed_outputs()

        m = tf.tile(self.mean_function(X), [1, self.num_task])
        m_observed = tf.boolean_mask(tf.reshape(m, [-1, 1]), observed_entry, axis=0)

        L = tf.linalg.cholesky(ks_observed)

        # [N*P,] log-likelihoods for of Y, all output channels considered
        log_prob = multivariate_normal(Y_observed, m_observed, L)
        """
        d = Y_observed - m_observed
        num_dims = tf.cast(tf.shape(d)[0], ks_observed.dtype)

        ks_inv = tf.linalg.inv(ks_observed)

        log_prob = -0.5 * tf.transpose(d) @ ks_inv @ d
        log_prob -= 0.5 * num_dims * np.log(2 * np.pi)
        log_prob -= 0.5 * tf.linalg.logdet(ks_observed)
        """
        return tf.reduce_sum(log_prob)

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        r"""
        make sure the kernel is a MultioutputKernel
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        if not isinstance(self.kernel, MultioutputKernel):
            raise ValueError(
                "this method currently only supports MultioutputKernel, make sure the model has the correct kernel object"
            )
        X_data, Y_data = self._aggregate_data()

        kmm, Y_observed, observed_entry = self._return_KNN_with_observed_outputs()
        m = tf.tile(self.mean_function(X_data), [1, self.num_task])
        m_observed = tf.boolean_mask(tf.reshape(m, [-1, 1]), observed_entry, axis=0)
        err = Y_observed - m_observed
        """
        L = cholesky(K) -> K = L @ L.T -> K^-1 = L.T^-1 @ L^-1
        so B.T @ K^-1 @ B = A.T @ A, where A = L^-1 @ B
        """
        kmn = self.kernel(
            X_data, Xnew, full_cov=True, full_output_cov=True
        )  # full_output_cov is default to True anyway, just to avoid confusion
        M, P, N, _ = tf.shape(kmn)
        kmn = tf.reshape(kmn, [M * P, N * P])
        kmn = tf.boolean_mask(kmn, observed_entry, axis=0)  # [MP-#unobserved, NP]
        knn = self.kernel(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)

        # now we have everything, compute predictive mean & cov
        # I don't use gpflow.conditionals.base_condition,
        # because my computation is a 1-D gp while knn has a P-dim shape (full_cov, full_output_cov should be taken care)

        # calculating full [N, P, N, P] is impossible as N grows
        # calculate only what we need
        Lm = tf.linalg.cholesky(kmm)
        A = tf.linalg.triangular_solve(Lm, kmn, lower=True)  # [MP-#unob, NP]

        if not full_cov and not full_output_cov:  # return [N, P]
            A_res = tf.reshape(A, [-1, N, P])
            K_Kinv_K = tf.reduce_sum(tf.square(A_res), axis=-3)
        elif not full_cov and full_output_cov:  # return [N, P, P]
            A_res = tf.reshape(A, [-1, N, P])
            K_Kinv_K = tf.einsum("...mnp, ...mnq -> ...npq", A_res, A_res)
        elif full_cov and not full_output_cov:  # return [P, N, N]
            A_res = tf.reshape(A, [-1, N, P])
            K_Kinv_K = tf.einsum("...mnp, ...mkp -> ...pnk", A_res, A_res)
        else:  # return [N, P, N, P]
            K_Kinv_K = tf.matmul(A, A, transpose_a=True)  # [NP, NP]
            K_Kinv_K = tf.reshape(K_Kinv_K, [N, P, N, P])

        f_var = knn - K_Kinv_K

        # compute functional mean
        # note: A = Lm^-1 @ kmn, shape [MP-#unob, NP]
        A = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm), A, lower=False)
        # now A = Lm^H^-1 @ Lm^-1 @ kmn, shape [MP-#unob, NP]
        A = tf.reshape(tf.transpose(A), [N, P, -1])
        f_mean_zero = tf.squeeze(tf.linalg.matmul(A, err), axis=-1)

        f_mean = f_mean_zero + self.mean_function(Xnew)

        return f_mean, f_var
