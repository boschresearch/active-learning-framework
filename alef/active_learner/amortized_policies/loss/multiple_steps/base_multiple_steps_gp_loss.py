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


import torch
from torch.nn.parallel.scatter_gather import scatter, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch._utils import _get_all_device_indices

from pyro.util import warn_if_nan

from ..base_loss import BaseLoss
from alef.active_learner.amortized_policies.utils.gp_computers import GaussianProcessComputer

from enum import Enum


class ProcessMethod(Enum):
    process = 0
    validation = 1


class _GPRMSEComputer(GaussianProcessComputer):
    def forward(self, kernel_list, noise_var_list, X, Y, X_test, Y_test):
        """
        :param kernel_list: a list of Nk gpytorch kernel on a single device
        :param noise_var_list: a list of Nk observation noise variance on a single device
        :param X: [B, Nk, Nf, T, D] tensor on a single device
        :param Y: [B, Nk, Nf, T] tensor on a single device
        :param X_test: [B, Nk, Nf, n_test, D] tensor on a single device
        :param Y_test: [B, Nk, Nf, n_test] tensor on a single device

        :return: [B, Nk, Nf] tensor of GP RMSEs on a single device
        """
        ###
        ### first compute p(f|X_test, X, Y)
        ### then compare Y_test to the mean of the posterior
        ###
        K_prior = self.compute_kernel_batch(
            kernel_list, X, X, noise_var_list=noise_var_list, batch_dim=1, gram_size_dim=-2
        )  # [B, num_kernels, num_functions, n_test, T]
        K_cross = self.compute_kernel_batch(
            kernel_list, X, X_test, batch_dim=1
        )  # [B, num_kernels, num_functions, T, n_test]

        mu_conditional = self.compute_gaussian_process_posterior(
            K_prior.to_dense(), K_cross.to_dense(), cov_test=None, Y_observation=Y, return_mu=True, return_cov=False
        )  # [B, num_kernels, num_functions, n_test]
        rmse = (mu_conditional - Y_test).pow(2).mean(dim=-1).sqrt()  # [B, num_kernels, num_functions]

        return rmse


class BaseMultipleStepsGPLoss(BaseLoss):
    def __init__(self, batch_size, num_kernels, num_functions_per_kernel, data_source=None, name=None):
        super().__init__(batch_size=batch_size, data_source=data_source, name=name)
        self.num_kernels = num_kernels
        self.num_functions_per_kernel = num_functions_per_kernel
        self.loss_computer = None
        self.test_computer = _GPRMSEComputer()
        """
        we later write main computation in the loss_computer.forward method
        """

    def compute_loss(self, process, process_kwargs):
        """
        compute the loss which needs to be differentiated
        """
        base_device = torch.device(process.device)
        process_return_list = self._execute_experiment(process, process_kwargs)
        loss = self._apply_computer_to_gp_rollout(
            self.loss_computer, process_return_list, base_device
        )  # [B, num_kernels, num_functions]

        loss = loss.mean()  # average over different batch samples & prior functions
        warn_if_nan(loss, "loss")
        return loss

    def validation(self, process):
        """
        compute the validation values which does not need to be differentiated (for visualization, analysis etc)

        return tuple of 2 floats, the mean and standard error of the validation values
        """
        B = min(self.batch_size, 10)
        Nk = min(self.num_kernels, 10)
        Nf = min(self.num_functions_per_kernel, 10)

        base_device = torch.device(process.device)
        process.design_net.eval()
        process_return_list = self._execute_experiment(
            process,
            {
                "batch_size": B,
                "num_kernels": Nk,
                "num_functions": Nf,
                "sample_domain_grid_points": True,
                "num_grid_points": 200,
            },
        )
        rmse = self._apply_computer_to_gp_rollout(
            self.test_computer, process_return_list, base_device
        )  # [B, num_kernels, num_functions]
        process.design_net.train()
        rmse_flatten = rmse.flatten()  # [B * Nk * Nf]

        rmse_mean = rmse_flatten.mean()
        rmse_stderr = torch.sqrt(rmse_flatten.var() / rmse_flatten.shape[0])
        return rmse_mean, rmse_stderr

    def _execute_experiment(self, process, process_kwargs, graph_type="flat", detach=False):
        r"""
        return a list, number of list == number of devices
        each element is the tuple of process return
        """
        assert "num_kernels" in process_kwargs
        assert not "device" in process_kwargs
        base_device = torch.device(process.device)
        n_devices = torch.cuda.device_count()
        if base_device.type == "cuda" and n_devices > 1:
            # run process(**process_kwargs) on multiple devices
            # separate on different kernels
            pass_in_process_kwargs = process_kwargs.copy()
            num_kernels = pass_in_process_kwargs.pop("num_kernels")

            device_ids = _get_all_device_indices()
            kernel_batch_template = torch.empty(num_kernels, device=base_device)
            kernel_batch_template = scatter(
                (kernel_batch_template,), device_ids, dim=0
            )  # kernel batch sizes on individual devices

            process_list = process.replicate(
                device_ids, not torch.is_grad_enabled()
            )  # clone process (including design_net, gp sampler) onto individual devices
            inputs = tuple(tuple() for _ in process_list)
            kwargs_list = [
                {
                    **pass_in_process_kwargs,
                    "num_kernels": kbt_d[0].shape[0],
                    "device": kbt_d[0].device,
                }
                for d, kbt_d in enumerate(kernel_batch_template)
            ]
            return parallel_apply(
                process_list, inputs, kwargs_list, device_ids[: len(process_list)]
            )  # run process(*input, **kwargs) on individual devices
        else:
            return [process(**process_kwargs)]

    def _apply_computer_to_gp_rollout(self, computer, process_return_list, output_device):
        r"""
        compute loss or validation values using the given computer module
        process datasets parallelly on different devices if possible
        return computation result on the original device
        """
        n_devices = len(process_return_list)
        if n_devices == 1:
            return computer(*process_return_list[0])  # [B, num_kernels, num_functions]
        else:
            device_ids = range(n_devices)
            replicas = replicate(computer, device_ids, not torch.is_grad_enabled())
            outputs = parallel_apply(replicas, process_return_list, None, device_ids)
            return gather(outputs, output_device, dim=1)  # [B, num_kernels, num_functions]
