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

from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.kernels.multi_output_kernels.latent_kernel_enum import LatentKernel
from alef.configs.base_parameters import (
    BASE_KERNEL_VARIANCE,
    BASE_KERNEL_LENGTHSCALE,
)


class BasicCoregionalization1LConfig(BaseKernelConfig):
    base_variance: float = BASE_KERNEL_VARIANCE
    base_lengthscale: float = BASE_KERNEL_LENGTHSCALE
    W_rank: int = 2
    input_dimension: int
    output_dimension: int
    add_prior: bool = False
    lengthscale_prior_parameters = (1, 9)  # gamma mean
    variance_prior_parameters = (1, 0.3)  # truncated normal
    latent_kernel: LatentKernel = LatentKernel.MATERN52
    active_on_single_dimension: bool = False
    active_dimension: int = None
    name: str = "BasicCoregion1L"


class Coregionalization1LWithPriorConfig(BasicCoregionalization1LConfig):
    add_prior: bool = True
    name = "Coregion1LWithPrior"
