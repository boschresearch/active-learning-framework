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
from alef.configs.kernels.base_elementary_kernel_config import BaseElementaryKernelConfig
from alef.configs.base_parameters import (
    BASE_KERNEL_VARIANCE,
    BASE_KERNEL_LENGTHSCALE,
    BASE_KERNEL_PERIOD,
)
from alef.configs.prior_parameters import (
    KERNEL_LENGTHSCALE_GAMMA,
    KERNEL_VARIANCE_GAMMA,
    PERIODIC_KERNEL_PERIOD_GAMMA,
)


class BasicPeriodicConfig(BaseElementaryKernelConfig):
    base_lengthscale: float = BASE_KERNEL_LENGTHSCALE
    base_variance: float = BASE_KERNEL_VARIANCE
    base_period: float = BASE_KERNEL_PERIOD
    add_prior: bool = False
    fix_lengthscale: bool = False
    fix_period: bool = False
    fix_variance: bool = False
    lengthscale_prior_parameters: Tuple[float, float] = KERNEL_LENGTHSCALE_GAMMA
    variance_prior_parameters: Tuple[float, float] = KERNEL_VARIANCE_GAMMA
    period_prior_parameters: Tuple[float, float] = PERIODIC_KERNEL_PERIOD_GAMMA
    name = "BasicPeriodicKernel"


class PeriodicWithPriorConfig(BasicPeriodicConfig):
    add_prior: bool = True
    name = "PeriodicKernelWithPrior"


class PeriodicWithPriorSmallerInitialPeriodicityConfig(PeriodicWithPriorConfig):
    base_period: float = 0.2
    name = "PeriodicWithPriorSmallerInitialPeriodicity"
