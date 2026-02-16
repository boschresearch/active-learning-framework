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

from abc import abstractmethod
from typing import Tuple, Union, Sequence
import gpytorch
import torch
import math
from gpytorch.constraints import Positive
from alef.kernels.pytorch_kernels.customized_gpytorch_kernels import PeriodicKernel


"""
ref:
Rahimi & Recht, NeurIPS 2007, Random Features for Large-Scale Kernel Machines
Wilson et al., ICML 2020, Efficiently Sampling Functions from Gaussian Process Posteriors
"""


class BayesianLinearModel(torch.nn.Module):
    def __init__(self, omega, bias, weight, x_expanded_already):
        super().__init__()
        self.register_buffer("omega", omega.clone())  # [num_kernels, num_functions, input_dim, L]
        self.register_buffer("bias", bias.clone())  # [num_kernels, num_functions, L]
        self.register_buffer("weight", weight.clone())  # [num_kernels, num_functions, L]
        self.x_expanded_already = x_expanded_already

    def forward(self, x: torch.Tensor):
        if self.x_expanded_already:
            # x: [B, num_kernels, num_functions, ..., input_dim] or [B, 1, 1, ..., input_dim]
            # num_kernels: batch size of kernel hyperparameters
            msg = f"shape of x[1:3] {x.shape[1:3]} does not match [num of kernel, num of funcs per kernel] [{self.omega.shape[0]}, {self.weight.shape[1]}]"
            assert x.shape[1] == 1 or x.shape[1] == self.omega.shape[0], msg
            assert x.shape[2] == 1 or x.shape[2] == self.weight.shape[1], msg
            xx = torch.einsum("bkf...->kfb...", x)  # turn to [num_kernels, num_functions, B, ..., input_dim]
            xx = xx.expand(self.omega.shape[:2] + xx.shape[2:]) if xx.flatten(0, 1).shape[0] == 1 else xx
        else:
            # x: [B, ..., input_dim]
            xx = x.expand(self.omega.shape[:2] + x.shape)  # turn to [num_kernels, num_functions, B, ..., input_dim]
        linear_opt_x = torch.einsum(
            "mnbd,mndl->mnbl", xx.flatten(start_dim=2, end_dim=-2), self.omega
        ) + self.bias.unsqueeze(-2)  # [num_kernels, num_functions, xx.shape[2:-1]_flatten, L]
        phi = math.sqrt(2.0 / self.weight.shape[1]) * torch.cos(
            linear_opt_x
        )  # [num_kernels, num_functions, xx.shape[2:-1]_flatten, L]
        f_out = torch.einsum("mnbl,mnl->mnb", phi, self.weight).reshape(
            xx.shape[:-1]
        )  # [num_kernels, num_functions, xx.shape[2:-1]]
        return torch.einsum("kfb...->bkf...", f_out)  # [B, num_kernels, num_functions, x.shape[3:-1]]


class BaseElementaryKernelPytorch(gpytorch.kernels.Kernel):
    has_fourier_feature = False

    def __init__(
        self,
        input_dimension: int,
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        self.input_dimension = input_dimension
        self.active_dimension = active_dimension
        self.active_on_single_dimension = active_on_single_dimension

        if active_on_single_dimension:
            self.name = name + "_on_" + str(active_dimension)
            super().__init__(active_dims=torch.tensor([active_dimension]))
            self.num_active_dimensions = 1
        else:
            self.name = name
            super().__init__()
            self.num_active_dimensions = input_dimension
        self.kernel = None

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        return self.kernel.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)

    def get_input_dimension(self):
        return self.input_dimension

    @abstractmethod
    def get_parameters_flattened(self) -> torch.tensor:
        raise NotImplementedError

    @property
    def prior_scale(self):  # this would work for matern family, RQ kernel, linear kernel, spectral mixture kernel
        D = self.get_input_dimension()
        dumpy_point = torch.ones([1, D], device=self.kernel.device)

        var_scale = self(dumpy_point, diag=True).evaluate().squeeze(-1)  # size is kernel batch size
        std_scale = torch.sqrt(var_scale)
        return std_scale

    ### the followings are for fourier features
    ### the followings are for fourier features
    ### the followings are for fourier features
    def _sample_feature_frequencies(self, L: int, num_functions: int = 1):
        r"""
        :param L: int, number of fourier features, see [1] page 3, this is the D in the paper
        :param num_functions: int, number of functions to sample (for batch purpose)

        return:
        torch.Tensor of shape [..., num_functions, input_dim, L], see [1] page 3, this is the omega in the paper

        [1] Rahimi & Recht, NeurIPS 2007, Random Features for Large-Scale Kernel Machines
        """
        if self.has_fourier_feature:
            raise NotImplementedError
        else:
            pass

    def _sample_feature_weights(self, L: int, num_functions: int = 1):
        r"""
        :param L: int, number of fourier features, see [1] page 3, this is the D in the paper
        :param num_functions: int, number of functions to sample (for batch purpose)

        return:

        torch.Tensor of shape [..., num_functions, L], see [1] page 3 and [2] page 3, this is the w in paper [2]

        notice that, if we sample w as in paper [2], then cov(f(x), f(x')) = 1 * <phi(x), phi(x') >

        [1] Rahimi & Recht, NeurIPS 2007, Random Features for Large-Scale Kernel Machines
        [2] Wilson et al., ICML 2020, Efficiently Sampling Functions from Gaussian Process Posteriors
        """
        if self.has_fourier_feature:
            raise NotImplementedError
        else:
            pass

    def sample_fourier_features(self, L: int, num_functions: int = 1):
        r"""
        :param L: int, number of fourier features, see [1] page 3, this is the D in the paper
        :param num_functions: int, number of functions to sample (for batch purpose)

        sample random fourier features
        see [2], page 3 for details

        [1] Rahimi & Recht, NeurIPS 2007, Random Features for Large-Scale Kernel Machines
        [2] Wilson et al., ICML 2020, Efficiently Sampling Functions from Gaussian Process Posteriors
        """
        if not self.has_fourier_feature:
            raise NotImplementedError
        else:
            self._num_functions = num_functions
            self._num_fourier_sample = L

            self._omega = self._sample_feature_frequencies(
                self._num_fourier_sample, num_functions
            )  # [..., num_functions, input_dim, L]
            self._bias = (
                2
                * math.pi
                * torch.rand(size=self._omega.shape[:-2] + (self._num_fourier_sample,)).to(self._omega.device)
            )  # [..., num_functions, L]
            self._weight = self._sample_feature_weights(
                self._num_fourier_sample, num_functions
            )  # [..., num_functions, L]

    def bayesian_linear_model(self, x_expanded_already: bool = False):
        r"""
        :param x_expanded_already: bool, whether the input x is already expanded to
                [B, num_kernels, num_functions, ..., D] or [B, 1, 1, ..., D]

        return Bayesian linear model f as a function
        see [1] and page 3 of [2] for details

        [1] Rahimi & Recht, NeurIPS 2007, Random Features for Large-Scale Kernel Machines
        [2] Wilson et al., ICML 2020, Efficiently Sampling Functions from Gaussian Process Posteriors
        """
        if not self.has_fourier_feature:
            raise NotImplementedError
        else:
            assert_msg = "no fourier samples, please first make samples (method 'sample_fourier_features(number_of_fourier_features)')"
            assert hasattr(self, "_weight"), assert_msg
            assert hasattr(self, "_omega"), assert_msg
            assert hasattr(self, "_bias"), assert_msg
            return BayesianLinearModel(self._omega, self._bias, self._weight, x_expanded_already)


class RBFKernelPytorch(BaseElementaryKernelPytorch):
    has_fourier_feature = True

    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: Union[float, Sequence[float]],
        base_variance: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        a_variance, b_variance = variance_prior_parameters
        if add_prior:
            lengthscale_prior = gpytorch.priors.GammaPrior(a_lengthscale, b_lengthscale)
            outputscale_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            rbf_kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=self.num_active_dimensions, lengthscale_prior=lengthscale_prior
            )
            self.kernel = gpytorch.kernels.ScaleKernel(rbf_kernel, outputscale_prior=outputscale_prior)
        else:
            rbf_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.num_active_dimensions)
            self.kernel = gpytorch.kernels.ScaleKernel(rbf_kernel)

        if not hasattr(base_lengthscale, "__len__"):
            lengthscales = torch.full((1, self.num_active_dimensions), base_lengthscale)
        else:
            lengthscales = torch.tensor(base_lengthscale).squeeze().unsqueeze(-2)
            assert lengthscales.shape[-1] == self.num_active_dimensions, (
                f"number of base_lengthscale '{lengthscales.shape[-1]}' does not match input dimension '{self.num_active_dimensions}'"
            )

        rbf_kernel.lengthscale = lengthscales
        self.kernel.outputscale = torch.tensor(base_variance)

    @property
    def prior_scale(self):  # this would work for matern family, RQ kernel, linear kernel, spectral mixture kernel
        return torch.sqrt(self.kernel.outputscale)  #  # size is kernel batch size

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        lengthscales_flattened = torch.flatten(self.kernel.base_kernel.lengthscale)
        if hasattr(self.kernel.outputscale, "__len__"):
            variance_flattened = torch.flatten(self.kernel.outputscale)
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.outputscale]))

        if sqrt_variance:
            variance_flattened = torch.sqrt(variance_flattened)

        return torch.concat((lengthscales_flattened, variance_flattened))

    ### the followings are for fourier features
    ### the followings are for fourier features
    ### the followings are for fourier features
    def _sample_feature_frequencies(self, L: int, num_functions: int = 1):
        # lengthscale shape: [1, input_dim] or [batch_size, 1, input_dim]
        lghscale = self.kernel.base_kernel.lengthscale
        device = lghscale.device
        if len(lghscale.shape) > 2:
            kernel_batch_size = lghscale.shape[:-2]
        else:
            kernel_batch_size = torch.Size([1])
        omega = torch.randn(size=kernel_batch_size + (num_functions, L, self.num_active_dimensions), device=device).div(
            lghscale[..., None, :, :]
        )
        return torch.transpose(omega, -1, -2)

    def _sample_feature_weights(self, L: int, num_functions: int = 1):
        # outputscale shape: no shape (scalar) or [batch_size, ]
        scale = self.kernel.outputscale
        device = scale.device
        if len(scale.shape) >= 1:
            kernel_batch_size = scale.shape
        else:
            kernel_batch_size = torch.Size([1])
        weights = torch.sqrt(scale.reshape(kernel_batch_size + (1, 1))) * torch.randn(
            size=kernel_batch_size + (num_functions, L), device=device
        )  # ~ N(0, kernel_variance)
        return weights


class Matern52KernelPytorch(BaseElementaryKernelPytorch):
    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: Union[float, Sequence[float]],
        base_variance: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        a_variance, b_variance = variance_prior_parameters
        if add_prior:
            lengthscale_prior = gpytorch.priors.GammaPrior(a_lengthscale, b_lengthscale)
            outputscale_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            matern_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=self.num_active_dimensions, lengthscale_prior=lengthscale_prior
            )
            self.kernel = gpytorch.kernels.ScaleKernel(matern_kernel, outputscale_prior=outputscale_prior)
        else:
            matern_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.num_active_dimensions)
            self.kernel = gpytorch.kernels.ScaleKernel(matern_kernel)

        if not hasattr(base_lengthscale, "__len__"):
            lengthscales = torch.full((1, self.num_active_dimensions), base_lengthscale)
        else:
            lengthscales = torch.tensor(base_lengthscale).squeeze().unsqueeze(-2)
            assert lengthscales.shape[-1] == self.num_active_dimensions, (
                f"number of base_lengthscale '{lengthscales.shape[-1]}' does not match input dimension '{self.num_active_dimensions}'"
            )

        matern_kernel.lengthscale = lengthscales
        self.kernel.outputscale = torch.tensor(base_variance)

    @property
    def prior_scale(self):  # this would work for matern family, RQ kernel, linear kernel, spectral mixture kernel
        return torch.sqrt(self.kernel.outputscale)  #  # size is kernel batch size

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        lengthscales_flattened = torch.flatten(self.kernel.base_kernel.lengthscale)
        if hasattr(self.kernel.outputscale, "__len__"):
            variance_flattened = torch.flatten(self.kernel.outputscale)
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.outputscale]))

        if sqrt_variance:
            variance_flattened = torch.sqrt(variance_flattened)

        return torch.concat((lengthscales_flattened, variance_flattened))


class Matern32KernelPytorch(BaseElementaryKernelPytorch):
    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: Union[float, Sequence[float]],
        base_variance: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        a_variance, b_variance = variance_prior_parameters
        if add_prior:
            lengthscale_prior = gpytorch.priors.GammaPrior(a_lengthscale, b_lengthscale)
            outputscale_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            matern_kernel = gpytorch.kernels.MaternKernel(
                nu=1.5, ard_num_dims=self.num_active_dimensions, lengthscale_prior=lengthscale_prior
            )
            self.kernel = gpytorch.kernels.ScaleKernel(matern_kernel, outputscale_prior=outputscale_prior)
        else:
            matern_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=self.num_active_dimensions)
            self.kernel = gpytorch.kernels.ScaleKernel(matern_kernel)

        if not hasattr(base_lengthscale, "__len__"):
            lengthscales = torch.full((1, self.num_active_dimensions), base_lengthscale)
        else:
            lengthscales = torch.tensor(base_lengthscale).squeeze().unsqueeze(-2)
            assert lengthscales.shape[-1] == self.num_active_dimensions, (
                f"number of base_lengthscale '{lengthscales.shape[-1]}' does not match input dimension '{self.num_active_dimensions}'"
            )

        matern_kernel.lengthscale = lengthscales
        self.kernel.outputscale = torch.tensor(base_variance)

    @property
    def prior_scale(self):  # this would work for matern family, RQ kernel, linear kernel, spectral mixture kernel
        return torch.sqrt(self.kernel.outputscale)  #  # size is kernel batch size

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        lengthscales_flattened = torch.flatten(self.kernel.base_kernel.lengthscale)
        if hasattr(self.kernel.outputscale, "__len__"):
            variance_flattened = torch.flatten(self.kernel.outputscale)
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.outputscale]))

        if sqrt_variance:
            variance_flattened = torch.sqrt(variance_flattened)

        return torch.concat((lengthscales_flattened, variance_flattened))


class PeriodicKernelPytorch(BaseElementaryKernelPytorch):
    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: Union[float, Sequence[float]],
        base_variance: float,
        base_period: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        period_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        a_period, b_period = period_prior_parameters
        a_variance, b_variance = variance_prior_parameters
        if add_prior:
            lengthscale_prior = gpytorch.priors.GammaPrior(a_lengthscale, b_lengthscale)
            period_prior = gpytorch.priors.GammaPrior(a_period, b_period)
            outputscale_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            periodic_kernel = PeriodicKernel(
                ard_num_dims=self.num_active_dimensions,
                period_length_prior=period_prior,
                lengthscale_prior=lengthscale_prior,
            )
            # periodic_kernel.register_prior(
            #    "lengthscale_prior",
            #    lengthscale_prior,
            #    lambda m: torch.sqrt(m.lengthscale) / 2.0,
            #    lambda m, v: m._set_lengthscale(torch.square(v) * 4.0),
            # )
            self.kernel = gpytorch.kernels.ScaleKernel(periodic_kernel, outputscale_prior=outputscale_prior)
        else:
            periodic_kernel = PeriodicKernel(ard_num_dims=self.num_active_dimensions)
            self.kernel = gpytorch.kernels.ScaleKernel(periodic_kernel)

        if not hasattr(base_lengthscale, "__len__"):
            lengthscales = torch.full((1, self.num_active_dimensions), base_lengthscale)
        else:
            lengthscales = torch.tensor(base_lengthscale).squeeze().unsqueeze(-2)
            assert lengthscales.shape[-1] == self.num_active_dimensions, (
                f"number of base_lengthscale '{lengthscales.shape[-1]}' does not match input dimension '{self.num_active_dimensions}'"
            )

        periods = torch.full((1, self.num_active_dimensions), base_period)
        periodic_kernel.lengthscale = lengthscales
        periodic_kernel.period_length = periods
        self.kernel.outputscale = base_variance

    @property
    def prior_scale(self):  # this would work for matern family, RQ kernel, linear kernel, spectral mixture kernel
        return torch.sqrt(self.kernel.outputscale)  #  # size is kernel batch size

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        lengthscales_flattened = torch.flatten(self.kernel.base_kernel.lengthscale)
        if hasattr(self.kernel.outputscale, "__len__"):
            variance_flattened = torch.flatten(self.kernel.outputscale)
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.outputscale]))

        if sqrt_variance:
            variance_flattened = torch.sqrt(variance_flattened)

        period_flattened = torch.flatten(self.kernel.base_kernel.period_length)

        return torch.concat((lengthscales_flattened, variance_flattened, period_flattened))


class RQKernelPytorch(BaseElementaryKernelPytorch):
    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: Union[float, Sequence[float]],
        base_variance: float,
        base_alpha: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        alpha_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        a_alpha, b_alpha = alpha_prior_parameters
        a_variance, b_variance = variance_prior_parameters
        if add_prior:
            lengthscale_prior = gpytorch.priors.GammaPrior(a_lengthscale, b_lengthscale)
            alpha_prior = gpytorch.priors.GammaPrior(a_alpha, b_alpha)
            outputscale_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            rq_kernel = gpytorch.kernels.RQKernel(
                ard_num_dims=self.num_active_dimensions, lengthscale_prior=lengthscale_prior
            )
            rq_kernel.register_prior(
                "alpha_prior",
                alpha_prior,
                lambda m: m.alpha,
                lambda m, v: m.initialize(raw_alpha=m.raw_alpha_constraint.inverse_transform(torch.to_tensor(v))),
            )
            self.kernel = gpytorch.kernels.ScaleKernel(rq_kernel, outputscale_prior=outputscale_prior)
        else:
            rq_kernel = gpytorch.kernels.RQKernel(ard_num_dims=self.num_active_dimensions)
            self.kernel = gpytorch.kernels.ScaleKernel(rq_kernel)

        if not hasattr(base_lengthscale, "__len__"):
            lengthscales = torch.full((1, self.num_active_dimensions), base_lengthscale)
        else:
            lengthscales = torch.tensor(base_lengthscale).squeeze().unsqueeze(-2)
            assert lengthscales.shape[-1] == self.num_active_dimensions, (
                f"number of base_lengthscale '{lengthscales.shape[-1]}' does not match input dimension '{self.num_active_dimensions}'"
            )

        rq_kernel.lengthscale = lengthscales
        rq_kernel.alpha = torch.tensor(base_alpha)
        self.kernel.outputscale = torch.tensor(base_variance)

    @property
    def prior_scale(self):  # this would work for matern family, RQ kernel, linear kernel, spectral mixture kernel
        return torch.sqrt(self.kernel.outputscale)  #  # size is kernel batch size

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        lengthscales_flattened = torch.flatten(self.kernel.base_kernel.lengthscale)
        if hasattr(self.kernel.outputscale, "__len__"):
            variance_flattened = torch.flatten(self.kernel.outputscale)
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.outputscale]))

        if sqrt_variance:
            variance_flattened = torch.sqrt(variance_flattened)

        alpha_flattened = torch.flatten(torch.tensor([self.kernel.base_kernel.alpha]))
        return torch.concat((lengthscales_flattened, variance_flattened, alpha_flattened))


class LinearKernelPytorch(BaseElementaryKernelPytorch):
    def __init__(
        self,
        input_dimension: int,
        base_variance: float,
        base_offset: float,
        add_prior: bool,
        variance_prior_parameters: Tuple[float, float],
        offset_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_variance, b_variance = variance_prior_parameters
        a_offset, b_offset = offset_prior_parameters
        if add_prior:
            variance_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            self.kernel = gpytorch.kernels.LinearKernel(
                num_dimensions=self.num_active_dimensions, variance_prior=variance_prior
            )
        else:
            self.kernel = gpytorch.kernels.LinearKernel(num_dimensions=self.num_active_dimensions)
        self.kernel.variance = torch.tensor(base_variance)

        self.register_parameter(name="raw_offset", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
        offset_constraint = Positive()

        self.register_constraint("raw_offset", offset_constraint)

        if add_prior:
            offset_prior = gpytorch.priors.GammaPrior(a_offset, b_offset)
            self.register_prior(
                "offset_prior",
                offset_prior,
                lambda m: m.offset,
                lambda m, v: m._set_offset(v),
            )

        self.offset = base_offset

    @property
    def offset(self):
        return self.raw_offset_constraint.transform(self.raw_offset)

    @offset.setter
    def offset(self, value):
        return self._set_offset(value)

    def _set_offset(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_offset)
        self.initialize(raw_offset=self.raw_offset_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert not last_dim_is_batch
        K = self.kernel.forward(x1, x2, diag, last_dim_is_batch, **params) + self.offset
        return K

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        offset_flattened = torch.flatten(torch.tensor([self.offset]))
        if hasattr(self.kernel.variance, "__len__"):
            variance_flattened = torch.flatten(self.kernel.variance)
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.variance]))

        if sqrt_variance:
            variance_flattened = torch.sqrt(variance_flattened)

        return torch.concat((offset_flattened, variance_flattened))


if __name__ == "__main__":
    linear_kernel = LinearKernelPytorch(3, 1.0, 1.0, True, (1.0, 1.0), (1.0, 1.0), False, 0, "Linear")
    print(linear_kernel.offset)
    X = torch.randn((10, 3))
    K = linear_kernel.forward(X, X, diag=True)
    print(X.numpy())
    print(K.detach().numpy())

    rbf_kernel = RBFKernelPytorch(2, [0.2, 0.4], 1.0, False, (1.0, 1.0), (1.0, 1.0), False, 0, "RBF")
    print(rbf_kernel.kernel.outputscale)
    print(rbf_kernel.kernel.base_kernel.lengthscale)

    from matplotlib import pyplot as plt

    Q = 5
    D = 1
    num_func = 5
    kernel = RBFKernelPytorch(D, 0.5, 1.0, False, (1, 1), (1, 1), False, 0, "RBF")
    print(kernel.get_parameters_flattened())
    kernel.sample_fourier_features(50, num_func)
    f = kernel.bayesian_linear_model()

    X = torch.randn([100, D])
    Y_rff = f(X).cpu().detach().squeeze()
    Y_gp = (
        torch.distributions.MultivariateNormal(
            torch.zeros(100), covariance_matrix=kernel(X).evaluate() + 0.0004 * torch.eye(100)
        )
        .sample([num_func])
        .cpu()
        .detach()
    )
    print(X.shape, Y_rff.shape, Y_gp.shape)

    fig, ax = plt.subplots(1, 2)
    for i in range(num_func):
        ax[0].plot(X, Y_rff[i], ".")
        ax[1].plot(X, Y_gp[i], ".")
    ax[0].set_title("rff")
    ax[1].set_title("gp")
    plt.show()
