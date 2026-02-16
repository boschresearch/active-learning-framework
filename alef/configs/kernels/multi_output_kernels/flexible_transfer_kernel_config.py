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

from typing import List
from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.kernels.multi_output_kernels.latent_kernel_enum import LatentKernel
from alef.configs.base_parameters import (
    BASE_KERNEL_VARIANCE,
    BASE_KERNEL_LENGTHSCALE,
)


class BasicFlexibleTransferConfig(BaseKernelConfig):
    variance_list: List = [BASE_KERNEL_VARIANCE, 0.001 * BASE_KERNEL_VARIANCE, BASE_KERNEL_VARIANCE]
    lengthscale_list: List = [BASE_KERNEL_LENGTHSCALE] * 3
    input_dimension: int
    output_dimension: int = 2
    add_prior: bool = False
    lengthscale_prior_parameters = (1, 9)  # gamma mean
    variance_prior_parameters = (1, 0.3)  # truncated normal
    latent_kernel: LatentKernel = LatentKernel.MATERN52
    active_on_single_dimension: bool = False
    active_dimension: int = None
    name: str = "BasicFlexibleTransferKernelConfig"


class FlexibleTransferWithPriorConfig(BasicFlexibleTransferConfig):
    add_prior: bool = True
    name = "FlexibleTransferKernelWithPrior"
