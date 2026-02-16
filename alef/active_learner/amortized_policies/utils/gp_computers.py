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

import math
import torch
from .utils import compute_kernel_batch


class GaussianProcessComputer(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_kernel_batch(
        self,
        kernel_list,
        x1,
        x2,
        noise_var_list=None,
        *,
        batch_dim: int = 1,
        gram_size_dim: int = -2,
        return_linear_operator: bool = False,
    ):
        return compute_kernel_batch(
            kernel_list,
            x1,
            x2,
            noise_var_list,
            batch_dim=batch_dim,
            gram_size_dim=gram_size_dim,
            return_linear_operator=return_linear_operator,
        )

    def compute_gaussian_entropy(self, cov_matrix):
        r"""
        :param cov_matrix: [*batch_shape, T, T] tensor
        :return: tensor of batch_shape, 1/2 * logdet( cov_matrix * 2*pi*e )
        """
        # return 1/2 * torch.logdet( cov_matrix * 2*math.pi*math.e )
        # torch.linalg.cholesky seems numerically more stable than torch.logdet
        cholesky = torch.linalg.cholesky(cov_matrix * 2 * math.pi * math.e)
        return cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    def compute_gaussian_log_likelihood(self, Y, mu, cov_matrix):
        r"""
        :param Y: [*batch_shape, T] tensor
        :param mu: float or [*batch_shape, T] tensor
        :param cov_matrix: [*batch_shape, T, T] tensor
        :return: tensor of batch_shape, log N(Y|mu, cov_matrix)
        """
        cholesky = torch.linalg.cholesky(cov_matrix)
        transformed_Y = torch.linalg.solve_triangular(cholesky, (Y - mu).unsqueeze(-1), upper=False).squeeze(-1)

        half_log_det = (cholesky * math.sqrt(2 * math.pi)).diagonal(dim1=-2, dim2=-1).log().sum(-1)

        log_prob = -half_log_det - 1 / 2 * torch.pow(transformed_Y, 2).sum(-1)
        return log_prob

    def compute_gaussian_process_posterior(
        self,
        cov_observation_matrix,
        cov_cross_matrix,
        cov_test=None,
        Y_observation=None,
        *,
        return_mu: bool = True,
        return_cov: bool = True,
    ):
        r"""
        :param cov_observation_matrix: [*batch_shape, P, P] tensor
        :param cov_cross_matrix: float or [*batch_shape, P, T] tensor
        :param cov_test: [*batch_shape, T, T] tensor, can be None if return_cov is False
        :param Y_observation: [*batch_shape, P] tensor, can None if return_mu is False
        :param return_mu: bool, return posterior mean or not
        :param return_cov: bool, return posterior covariance or not
        :return: (mu, cov), or mu, or cov
            mu: tensor of [*batch_shape, T], cov_cross_matrix.T @ cov_observation_matrix^{-1} @ Y_observation
            cov: tensor of [*batch_shape, T, T], cov_test - cov_cross_matrix.T @ cov_observation_matrix^{-1} @ cov_cross_matrix
        """
        assert cov_cross_matrix.shape[-2] == cov_observation_matrix.shape[-2] == cov_observation_matrix.shape[-1]

        if not return_mu and not return_cov:
            raise ValueError("At least one of return_mu and return_cov must be True")

        else:
            assert cov_observation_matrix.shape[-1] == cov_observation_matrix.shape[-2] == cov_cross_matrix.shape[-2]
            cholesky = torch.linalg.cholesky(cov_observation_matrix)  # [*batch_shape, P, P]
            K_right = torch.linalg.solve_triangular(cholesky, cov_cross_matrix, upper=False)  # [*batch_shape, P, T]

            if return_mu:
                assert Y_observation is not None
                err = Y_observation
                transformed_Y_observation = torch.linalg.solve_triangular(
                    cholesky, err.unsqueeze(-1), upper=False
                )  # [*batch_shape, P, 1]

                mu_conditional = torch.matmul(K_right.transpose(-1, -2), transformed_Y_observation).squeeze(
                    -1
                )  # [*batch_shape, T]

            if return_cov:
                assert cov_test is not None
                assert cov_test.shape[-1] == cov_test.shape[-2] == cov_cross_matrix.shape[-1]

                K_conditional = cov_test - torch.matmul(K_right.transpose(-1, -2), K_right)  # [*batch_shape, T, T]

            if return_mu and not return_cov:
                return mu_conditional
            elif not return_mu and return_cov:
                return K_conditional
            else:
                return mu_conditional, K_conditional
