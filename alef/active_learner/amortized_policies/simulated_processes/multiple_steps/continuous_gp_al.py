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
from torch.nn.parallel.replicate import replicate
import pyro
import pyro.distributions as dist

from typing import Optional

from alef.active_learner.amortized_policies.utils.oed_primitives import (
    observation_sample,
    compute_design,
)
from alef.active_learner.amortized_policies.simulated_processes.multiple_steps.base_multiple_steps_process import (
    BaseMultipleStepsProcess,
)

# need GP model
from alef.configs.kernels.pytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig
from alef.active_learner.amortized_policies.simulated_processes.gp_sampler.rff_sampler import (
    RandomFourierFeatureSampler,
)


class SequentialGaussianProcessContinuousDomain(BaseMultipleStepsProcess):
    def __init__(
        self,
        design_net,
        kernel_config: BaseKernelPytorchConfig,
        n_steps: int = 20,
        sample_gp_prior: bool = True,
        # sample_method: GPSampleMethod = GPSampleMethod.RFF,
        device: torch.device = torch.device("cpu"),
    ):
        # make sure design_net is on the desired device
        super().__init__(design_net=design_net, n_steps=n_steps)
        # kernel = PytorchKernelFactory.build(kernel_config)
        self.gp_dist = RandomFourierFeatureSampler(kernel_config, 0.1, 1.01)
        self._sample_gp_prior = sample_gp_prior
        self.set_device(device)
        self.name = "Default"

    def replicate(self, device_ids, detach=False):
        design_net_list = replicate(self.design_net, device_ids, detach)
        new_list = []
        for d, new_net in enumerate(design_net_list):
            new = self.__new__(type(self))
            super(SequentialGaussianProcessContinuousDomain, new).__init__(new_net, self.n_steps)
            new.gp_dist = self.gp_dist.clone_module()
            new._sample_kernel = self._sample_gp_prior
            new.set_device(device_ids[d])
            new.name = f"Device_{d}"
            new_list.append(new)
        return new_list

    def set_device(self, device: torch.device):
        self.to(device)
        self.design_net.to(device)
        self.gp_dist.set_device(device)
        self.device = device

    def process(
        self,
        batch_size: int = 1,
        num_kernels: int = 1,
        num_functions: int = 1,
        sample_domain_grid_points: bool = False,
        num_grid_points: int = 100,
        device: Optional[torch.device] = None,
    ):
        r"""
        Generate a sequence of data. Each data point is an input-output pair,
        where inputs are experimental designs xi and outputs are observations y.
        This class provide observations sampled from GP simulators.

        :param batch_size: batch size.
        :param num_kernels: batch size of kernel hyperparameters (i.e. num of kernels).
        :param num_functions: number of functional sample given a GP prior.
        :param sample_domain_grid_points: if True, return some domain samples
                and their corresponding y values which are jointly GP with y
        :param num_grid_points: number of grid points per batch
        :param device: device to run the simulation on
        :return: simulation kernel, simulation noise_var, list of xi, list of y (, x_grid, y_grid)
        """
        if hasattr(self.design_net, "parameters"):
            #! this is required for the pyro optimizer
            pyro.module("design_net", self.design_net)

        assert num_kernels > 0
        assert num_functions > 0
        ########################################################################
        # Sample latent variables
        ########################################################################
        # Theta has empty shape
        self.gp_dist.draw_parameter(
            num_priors=num_kernels, num_functions=num_functions, draw_hyper_prior=self._sample_gp_prior
        )
        if device is not None:
            self.set_device(device)
        with torch.no_grad():
            kernel_list = self.gp_dist.kernel_list
            noise_var = self.gp_dist.noise_variance
        #######################################################################
        ### start sampling
        #######################################################################
        y_outcomes = []
        xi_designs = []

        for t in range(self.n_steps):
            ####################################################################
            # Get a design xi
            ####################################################################
            xi = compute_design(
                f"{self.name}_xi{t + 1}",
                self.design_net.lazy(*zip(xi_designs, y_outcomes)).expand((batch_size, num_kernels, num_functions)),
            )  # xi should have batch size [batch_size_for_each_function, num_kernels, num_functions, 1 (T placeholder), D]

            ####################################################################
            # Sample y
            ####################################################################
            pdf = self.gp_dist.y_sampler(xi).to_event(1)
            # [batch_size_for_each_function, num_kernels, num_functions, 1 (T placeholder)]
            y = observation_sample(f"{self.name}_y{t + 1}", pdf)
            xi_designs.append(xi)
            y_outcomes.append(y)

        if sample_domain_grid_points:
            D = self.gp_dist.kernel.input_dimension
            x_grid = (
                dist.Uniform(*self.input_domain)
                .sample((num_grid_points, D))
                .expand(xi.shape[:-2] + (num_grid_points, D))
            )
            with torch.no_grad():
                pdf = self.gp_dist.y_sampler(x_grid).to_event(1)
                y_grid = pdf.sample()
            return kernel_list, noise_var, torch.cat(xi_designs, dim=-2), torch.cat(y_outcomes, dim=-1), x_grid, y_grid
        else:
            return kernel_list, noise_var, torch.cat(xi_designs, dim=-2), torch.cat(y_outcomes, dim=-1)

    def validation(
        self,
        batch_size: int = 1,
        num_kernels: int = 1,
        num_functions: int = 1,
        num_test_points: int = 100,
        device: Optional[torch.device] = None,
    ):
        r"""
        Generate a sequence of data. Each data point is an input-output pair,
        where inputs are experimental designs xi and outputs are observations y.
        This class provide observations sampled from GP simulators.

        :param batch_size: batch size.
        :param num_kernels: batch size of kernel hyperparameters (i.e. num of kernels).
        :param num_functions: number of functional sample given a GP prior.
        :param num_grid_points: number of grid points per batch
        :param device: device to run the simulation on
        :return: simulation kernel, simulation noise_var, list of xi, list of y , x_test, y_test)
        """
        self.design_net.eval()
        out = self.process(batch_size, num_kernels, num_functions, True, num_test_points, device)
        self.design_net.train()
        return out


class PytestSequentialGaussianProcessContinuousDomain(SequentialGaussianProcessContinuousDomain):
    def process(self, *args, **kwargs):
        get_process = super().process(*args, **kwargs)
        if len(get_process) == 4:
            return get_process
        elif len(get_process) == 6:
            k_list, noise_var, xi, y, x_grid, y_grid = get_process
            return (
                k_list,
                noise_var,
                xi,
                y,
                torch.cat([torch.zeros_like(x_grid[..., :1, :]), torch.ones_like(x_grid[..., :1, :])], dim=-2),
                torch.ones_like(y_grid[..., :2]),
            )
