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
import gpytorch
from copy import deepcopy
from pyro.distributions import Normal, Uniform, Delta

from alef.active_learner.amortized_policies.simulated_processes.gp_sampler.base_sampler import BaseSampler


from alef.configs.kernels.pytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig
from alef.kernels.pytorch_kernels.pytorch_kernel_factory import PytorchKernelFactory


class RandomFourierFeatureSampler(BaseSampler):
    def __init__(
        self, kernel_config: BaseKernelPytorchConfig, observation_noise: float, overall_variance: float = 1.01
    ):
        r"""
        :param kernel_config: BaseKernelPytorchConfig
        :param observation_noise: float, observation noise standard deviation
        :param overall_variance: float, overall variance of the GP (variance scale of kernel + noise variance)
        """
        super().__init__()
        assert overall_variance > 1.0, "we assume the overall variance is greater than 1.0"
        self.__kernel_config = kernel_config
        self.register_buffer("_original_noise_variance", torch.tensor(math.pow(observation_noise, 2.0)))
        self.register_buffer("noise_variance", self._original_noise_variance.clone())
        self.register_buffer("_overall_variance", torch.tensor(overall_variance))
        # self._original_noise_variance = torch.tensor(math.pow(observation_noise, 2.0), device=device)
        # self.noise_variance = deepcopy(self._original_noise_variance)
        # self.noise_prior = Gamma(torch.tensor(1.0, device=device), 1 / torch.pow(torch.tensor(expected_observation_noise, device=device), 2.0))

        self._original_kernel = PytorchKernelFactory.build(kernel_config)
        self.kernel = deepcopy(self._original_kernel)
        assert self.kernel.has_fourier_feature
        self.kernel_list = [self.kernel.kernel]

        self._num_fourier_sample = 100
        self.draw_parameter(draw_hyper_prior=False)

    def clone_module(self):
        new = self.__new__(type(self))
        new.__init__(self.__kernel_config, torch.sqrt(self._original_noise_variance), self._overall_variance)
        return new

    def draw_parameter(self, num_priors: int = 1, num_functions: int = 1, draw_hyper_prior: bool = False):
        """
        draw hyper-priors, f, or noise of y|f

        arguments:

        num_priors: batch size of kernel hyperparameters (i.e. num of kernels).
        num_functions: number of functional sample given a GP prior.
        draw_hyper_prior: whether to draw parameters from hyper-priors.

        """
        assert num_priors > 0
        assert num_functions > 0
        if num_priors > 1:
            self.kernel.kernel = self._original_kernel.kernel.expand_batch((num_priors,))
            self.noise_variance = self._original_noise_variance.expand((num_priors,))
        else:  # reset to original
            self.kernel = deepcopy(self._original_kernel)
            self.noise_variance = deepcopy(self._original_noise_variance)

        if draw_hyper_prior:
            # I intentionally avoid gpytorch.kernel.pyro_sample_from_prior method
            # because we use pyro sample sites to handle the other part of our policy training,
            # if we do pyro sample here, our shapes will go wrong
            kernel = gpytorch.kernels.AdditiveKernel(self.kernel.kernel).to_random_module()
            # if we don't wrap the kernel into gpytorch.kernel.AdditiveKernel,
            # after setting_closure, somehow the parameters are removed from
            # from the torch.nn.Module parameters list, which cause problems later when we index kernels
            for prior_name, module, prior, closure, setting_closure in kernel.named_priors():
                # value = prior.sample(closure(module).shape)
                if "outputscale" in prior_name or "variance" in prior_name:
                    low = 0.505
                elif "offset" in prior_name:
                    low = -1.0
                else:
                    low = 0.05
                value = Uniform(low, torch.ones_like(closure(module))).sample()
                setting_closure(module, value)
            self.kernel.kernel = kernel.kernels[0]

            self.noise_variance = self._overall_variance * torch.ones_like(
                self.noise_variance
            ) - self.kernel.prior_scale.pow(2)  # + NOISE_VARIANCE_LOWER_BOUND

        if num_priors > 1:
            self.kernel_list = [self.kernel.kernel[i] for i in range(num_priors)]
        else:
            self.kernel_list = [self.kernel.kernel]

        self.kernel.sample_fourier_features(self._num_fourier_sample, num_functions)
        self._f_map = self.kernel.bayesian_linear_model(x_expanded_already=True)
        # wrap_parallel(
        #    self.kernel.bayesian_linear_model(x_expanded_already=True),
        #    x_torch.is_cuda,
        #    dim=0
        # )

    def f_sampler(self, x_data: torch.Tensor):
        """
        compute GP( mean(x_data), kernel(x_data) ), return in raw torch type

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        """
        x_torch = x_data
        f = self._f_map(x_torch)  # [batch_size, num_priors, num_functions]
        return Delta(f)  # .squeeze(2).squeeze(1)) # [batch_size] if num_priors == num_functions == 1

    def y_sampler(self, x_data: torch.Tensor):
        """
        compute GP( mean(x_data), kernel(x_data) ) + noise_dist, return in raw torch type

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        """
        x_torch = x_data
        f = self._f_map(x_torch)  # [batch_size, num_priors, num_functions]
        if f.shape[1] == 1:
            noise_std = torch.sqrt(self.noise_variance) * torch.ones_like(f)
        else:
            var_shape = [1] * f.dim()
            var_shape[1] = f.shape[1]
            var = self.noise_variance.reshape(var_shape)
            noise_std = torch.sqrt(var) * torch.ones_like(f)
        return Normal(
            f,  # .squeeze(2).squeeze(1),
            noise_std,  # .squeeze(2).squeeze(1)
        )  # [batch_size] if num_priors == num_functions == 1
