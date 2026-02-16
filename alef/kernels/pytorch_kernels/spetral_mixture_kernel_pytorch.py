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

from typing import Tuple
import gpytorch
import torch
import math
from alef.kernels.pytorch_kernels.elementary_kernels_pytorch import BaseElementaryKernelPytorch

"""
ref:
Rahimi & Recht, NeurIPS 2007, Random Features for Large-Scale Kernel Machines
Andrew G. Wilson & Ryan P. Adams, ICML 2013,
Gaussian Process Kernels for Pattern Discovery and Extrapolation
"""

# gpytorch.kernels.SpectralMixtureKernel does not support priors. So we have to first implement our own.


class _CustomizedSpectralMixtureKernel(gpytorch.kernels.SpectralMixtureKernel):
    pass


class SpectralMixtureKernelPytorch(BaseElementaryKernelPytorch):
    has_fourier_feature = True

    def __init__(
        self,
        num_mixtures: int,
        input_dimension: int,
        add_prior: bool,
        spectral_scale_prior_parameters: Tuple[float, float],
        spectral_mean_prior_parameters: Tuple[float, float],
        weight_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_spectral_scale, b_spectral_scale = spectral_scale_prior_parameters
        a_spectral_mean, b_spectral_mean = spectral_mean_prior_parameters
        a_weight, b_weight = weight_prior_parameters
        if add_prior:
            mixture_scales_prior = gpytorch.priors.GammaPrior(a_spectral_scale, b_spectral_scale)
            mixture_means_prior = gpytorch.priors.GammaPrior(a_spectral_mean, a_spectral_mean)
            # mixture means can be negative, but are the same as if they are positive, so we just focus on positive values
            # see paper eq. 12
            mixture_weights_prior = gpytorch.priors.GammaPrior(a_weight, b_weight)
            self.kernel = _CustomizedSpectralMixtureKernel(
                num_mixtures=num_mixtures,
                ard_num_dims=self.num_active_dimensions,
                mixture_scales_prior=mixture_scales_prior,
                mixture_means_prior=mixture_means_prior,
                mixture_weights_prior=mixture_weights_prior,
            )
        else:
            self.kernel = _CustomizedSpectralMixtureKernel(
                num_mixtures=num_mixtures, ard_num_dims=self.num_active_dimensions
            )

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        mixture_scales_flattened = torch.flatten(self.kernel.mixture_scales)
        mixture_means_flattened = torch.flatten(self.kernel.mixture_means)
        mixture_weights_flattened = torch.flatten(self.kernel.mixture_weights)
        return torch.concat((mixture_scales_flattened, mixture_means_flattened, mixture_weights_flattened))

    ### the followings are for fourier features
    ### the followings are for fourier features
    ### the followings are for fourier features
    def _sample_feature_frequencies(self, L: int, num_functions: int = 1):
        # mixture_weights shape: [Q] or [batch_size, Q]
        # mixture_scales shape: [Q, 1, D] or [batch_size, Q, 1, D]
        # mixture_means shape: [Q, 1, D] or [batch_size, Q, 1, D]
        mixture_weights = self.kernel.mixture_weights
        if len(mixture_weights.shape) >= 2:
            kernel_batch_size = mixture_weights.shape[:-1]
        else:
            kernel_batch_size = torch.Size([1])
        device = mixture_weights.device
        q = torch.multinomial(
            self.kernel.mixture_weights, num_samples=num_functions * L, replacement=True
        )  # [num_functions * L, ] or [batch_size, num_functions * L]
        q = q[..., None, None].expand(q.shape + (1, self.num_active_dimensions))
        mixture_scales = torch.gather(self.kernel.mixture_scales, -3, q).reshape(
            kernel_batch_size + (num_functions, L, self.num_active_dimensions)
        )
        mixture_means = torch.gather(self.kernel.mixture_means, -3, q).reshape(
            kernel_batch_size + (num_functions, L, self.num_active_dimensions)
        )
        flip = torch.randint(0, 2, size=kernel_batch_size + (num_functions, L), device=device) * 2 - 1
        mixture_means = mixture_means * flip.unsqueeze(-1)

        omega = (
            torch.randn(size=kernel_batch_size + (num_functions, L, self.num_active_dimensions), device=device)
            * math.sqrt(2)
            * math.pi
            * mixture_scales
            + mixture_means
        )
        return torch.transpose(omega, -1, -2)

    def _sample_feature_weights(self, L: int, num_functions: int = 1):
        # mixture_weights shape: [Q] or [batch_size, Q]
        mixture_weights = self.kernel.mixture_weights
        device = mixture_weights.device
        if len(mixture_weights.shape) > 1:
            kernel_batch_size = mixture_weights.shape[:-1]
        else:
            kernel_batch_size = torch.Size([1])
        weights = torch.randn(size=kernel_batch_size + (num_functions, L), device=device)  # ~ N(0, 1)
        return weights


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    Q = 5
    D = 1
    num_kernels = 2
    num_func = 5
    n_data = 500
    kernel = SpectralMixtureKernelPytorch(Q, D, False, (1, 1), (1, 1), (1, 1), False, 0, "SM")
    print(kernel.get_parameters_flattened())
    print(kernel.batch_shape)
    kernel.kernel = kernel.kernel.expand_batch((num_kernels,))
    print(kernel.get_parameters_flattened())
    print(kernel.batch_shape)
    kernel.kernel = kernel.kernel.expand_batch((num_kernels,))
    print(kernel.batch_shape)
    kernel.sample_fourier_features(50 * Q, num_func)
    f = kernel.bayesian_linear_model()

    X = torch.randn([n_data, D])
    Y_rff = f(X).cpu().detach().squeeze() + 0.1 * torch.randn([num_func, n_data])
    Y_gp = (
        torch.distributions.MultivariateNormal(
            torch.zeros(n_data), covariance_matrix=kernel(X).evaluate() + 0.01 * torch.eye(n_data)
        )
        .sample([num_func])
        .cpu()
        .detach()
    )
    print(X.shape, Y_rff.shape, Y_gp.shape)
    exit()

    fig, ax = plt.subplots(1, 2)
    for i in range(num_func):
        ax[0].plot(X, Y_rff[i], ".")
        ax[1].plot(X, Y_gp[i], ".")
    ax[0].set_title("rff")
    ax[1].set_title("gp")
    plt.show()
