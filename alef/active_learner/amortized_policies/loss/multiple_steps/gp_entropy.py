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


class BaseEntropy(BaseMultipleStepsGPLoss):
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
                "sample_domain_grid_points": False,
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


class _GPEntropyComputer1(GaussianProcessComputer):
    def forward(self, kernel_list, noise_var_list, X, Y):
        """
        :param kernel_list: a list of Nk gpytorch kernel on a single device
        :param noise_var_list: a list of Nk observation noise variance on a single device
        :param X: [B, Nk, Nf, T, D] tensor on a single device
        :param Y: [B, Nk, Nf, T] tensor on a single device

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
        return -entropy


class GPEntroopy1(BaseEntropy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_computer = _GPEntropyComputer1()


class _GPEntropyComputer2(GaussianProcessComputer):
    def forward(self, kernel_list, noise_var_list, X, Y):
        """
        :param kernel_list: a list of Nk gpytorch kernel on a single device
        :param noise_var_list: a list of Nk observation noise variance on a single device
        :param X: [B, Nk, Nf, T, D] tensor on a single device
        :param Y: [B, Nk, Nf, T] tensor on a single device

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
        return log_prob


class GPEntroopy2(BaseEntropy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_computer = _GPEntropyComputer2()
