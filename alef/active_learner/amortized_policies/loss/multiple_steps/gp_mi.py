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


from pyro.infer.util import torch_item

from .base_multiple_steps_gp_loss import BaseMultipleStepsGPLoss
from alef.active_learner.amortized_policies.utils.gp_computers import GaussianProcessComputer


class BaseMutualInformation(BaseMultipleStepsGPLoss):
    def differentiable_loss(self, process, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the design
        """
        return self.compute_loss(
            process,
            {
                "batch_size": self.batch_size,
                "num_kernels": self.num_kernels,
                "num_functions": self.num_functions_per_kernel,
                "sample_domain_grid_points": True,
            },
        )

    def loss(self, process, *args, **kwargs):
        """
        :returns: returns an estimate of the entropy
        :rtype: float
        Evaluates the minus entropy
        """
        loss_to_constant = torch_item(self.differentiable_loss(process, *args, **kwargs))
        return loss_to_constant


class _GPMutualInformationComputer1(GaussianProcessComputer):
    def forward(self, kernel_list, noise_var_list, X, Y, X_grid, Y_grid):
        """
        :param kernel_list: a list of Nk gpytorch kernel on a single device
        :param noise_var_list: a list of Nk observation noise variance on a single device
        :param X: [B, Nk, Nf, T, D] tensor on a single device
        :param Y: [B, Nk, Nf, T] tensor on a single device
        :param X_grid: [B, Nk, Nf, n_grid, D] tensor on a single device
        :param Y_grid: [B, Nk, Nf, n_grid] tensor on a single device

        :return: [B, Nk, Nf] tensor on a single device, where the mean should be the loss
        """
        K = self.compute_kernel_batch(
            kernel_list,
            X,
            X,
            noise_var_list=noise_var_list,
            batch_dim=1,
            gram_size_dim=-2,
            return_linear_operator=False,
        )  # [B, num_kernels, num_functions, T, T]
        entropy = self.compute_gaussian_entropy(K)  # [B, num_kernels, num_functions]

        #
        # now compute conditional entropy
        #
        K_grid = self.compute_kernel_batch(
            kernel_list,
            X_grid,
            X_grid,
            noise_var_list=noise_var_list,
            batch_dim=1,
            gram_size_dim=-2,
            return_linear_operator=False,
        )  # [B, num_kernels, num_functions, n_grid, n_grid]
        K_cross = self.compute_kernel_batch(
            kernel_list, X_grid, X, batch_dim=1, gram_size_dim=-2, return_linear_operator=False
        )  # [B, num_kernels, num_functions, n_grid, T]

        K_conditional = self.compute_gaussian_process_posterior(
            K_grid, K_cross, K, return_mu=False, return_cov=True
        )  # [B, num_kernels, num_functions, T, T]

        regularizer = self.compute_gaussian_entropy(K_conditional)  # [B, num_kernels, num_functions]

        neg_mi = regularizer - entropy  # [B, num_kernels, num_functions]
        return neg_mi


class GPMutualInformation1(BaseMutualInformation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_computer = _GPMutualInformationComputer1()


class _GPMutualInformationComputer2(GaussianProcessComputer):
    def forward(self, kernel_list, noise_var_list, X, Y, X_grid, Y_grid):
        """
        :param kernel_list: a list of Nk gpytorch kernel on a single device
        :param noise_var_list: a list of Nk observation noise variance on a single device
        :param X: [B, Nk, Nf, T, D] tensor on a single device
        :param Y: [B, Nk, Nf, T] tensor on a single device
        :param X_grid: [B, Nk, Nf, n_grid, D] tensor on a single device
        :param Y_grid: [B, Nk, Nf, n_grid] tensor on a single device

        :return: [B, Nk, Nf] tensor on a single device, where the mean should be the loss
        """
        K = self.compute_kernel_batch(
            kernel_list,
            X,
            X,
            noise_var_list=noise_var_list,
            batch_dim=1,
            gram_size_dim=-2,
            return_linear_operator=False,
        )  # [B, num_kernels, num_functions, T, T]
        log_prob = self.compute_gaussian_log_likelihood(Y, 0, K)  # [B, num_kernels, num_functions]

        #
        # now compute conditional log_prob
        #
        K_grid = self.compute_kernel_batch(
            kernel_list,
            X_grid,
            X_grid,
            noise_var_list=noise_var_list,
            batch_dim=1,
            gram_size_dim=-2,
            return_linear_operator=False,
        )  # [B, num_kernels, num_functions, n_grid, n_grid]
        K_cross = self.compute_kernel_batch(
            kernel_list, X_grid, X, batch_dim=1, gram_size_dim=-2, return_linear_operator=False
        )  # [B, num_kernels, num_functions, n_grid, T]

        mu_conditional, K_conditional = self.compute_gaussian_process_posterior(
            K_grid, K_cross, K, Y_observation=Y_grid, return_mu=True, return_cov=True
        )

        regularizer = self.compute_gaussian_log_likelihood(
            Y, mu_conditional, K_conditional
        )  # [B, num_kernels, num_functions]

        neg_mi = log_prob - regularizer  # [B, num_kernels, num_functions]
        return neg_mi


class GPMutualInformation2(BaseMutualInformation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_computer = _GPMutualInformationComputer2()
