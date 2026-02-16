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

from typing import Tuple, Union, Sequence
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import BaseElementaryKernelPytorchConfig
from alef.configs.base_parameters import (
    BASE_KERNEL_VARIANCE,
    BASE_KERNEL_LENGTHSCALE,
)
from alef.configs.prior_parameters import (
    KERNEL_LENGTHSCALE_GAMMA,
    KERNEL_VARIANCE_GAMMA,
    PERIODIC_KERNEL_PERIOD_GAMMA,
)


class BasicSpectralMixturePytorchConfig(BaseElementaryKernelPytorchConfig):
    base_lengthscale: Union[float, Sequence[float]] = BASE_KERNEL_LENGTHSCALE
    base_variance: float = BASE_KERNEL_VARIANCE
    add_prior: bool = False
    lengthscale_prior_parameters: Tuple[float, float] = KERNEL_LENGTHSCALE_GAMMA
    variance_prior_parameters: Tuple[float, float] = KERNEL_VARIANCE_GAMMA
    name = "BasicRBF"
    num_mixtures: int = 5
    add_prior: bool = False
    spectral_scale_prior_parameters: Tuple[float, float] = KERNEL_LENGTHSCALE_GAMMA
    spectral_mean_prior_parameters: Tuple[float, float] = PERIODIC_KERNEL_PERIOD_GAMMA
    weight_prior_parameters: Tuple[float, float] = KERNEL_VARIANCE_GAMMA
    name: str = "BasicSpectralMixture"


class SpectralMixtureWithPriorPytorchConfig(BasicSpectralMixturePytorchConfig):
    add_prior: bool = True
    name = "SpectralMixturewithPrior"
