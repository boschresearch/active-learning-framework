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
from alef.configs.kernels.pytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig
from alef.configs.base_parameters import (
    BASE_KERNEL_VARIANCE,
    BASE_KERNEL_LENGTHSCALE,
    BASE_LINEAR_KERNEL_OFFSET,
    BASE_RQ_KERNEL_ALPHA,
    BASE_KERNEL_PERIOD,
)
from alef.configs.prior_parameters import (
    KERNEL_LENGTHSCALE_GAMMA,
    KERNEL_VARIANCE_GAMMA,
    PERIODIC_KERNEL_PERIOD_GAMMA,
    LINEAR_KERNEL_OFFSET_GAMMA,
    RQ_KERNEL_ALPHA_GAMMA,
)


class BaseElementaryKernelPytorchConfig(BaseKernelPytorchConfig):
    active_on_single_dimension: bool = False
    active_dimension: int = 0


class BasicRBFPytorchConfig(BaseElementaryKernelPytorchConfig):
    base_lengthscale: Union[float, Sequence[float]] = BASE_KERNEL_LENGTHSCALE
    base_variance: float = BASE_KERNEL_VARIANCE
    add_prior: bool = False
    lengthscale_prior_parameters: Tuple[float, float] = KERNEL_LENGTHSCALE_GAMMA
    variance_prior_parameters: Tuple[float, float] = KERNEL_VARIANCE_GAMMA
    name = "BasicRBF"


class RBFWithPriorPytorchConfig(BasicRBFPytorchConfig):
    add_prior: bool = True
    name = "RBFwithPrior"


class BasicMatern52PytorchConfig(BaseElementaryKernelPytorchConfig):
    base_lengthscale: Union[float, Sequence[float]] = BASE_KERNEL_LENGTHSCALE
    base_variance: float = BASE_KERNEL_VARIANCE
    add_prior: bool = False
    lengthscale_prior_parameters: Tuple[float, float] = KERNEL_LENGTHSCALE_GAMMA
    variance_prior_parameters: Tuple[float, float] = KERNEL_VARIANCE_GAMMA
    name = "BasicMatern52"


class Matern52WithPriorPytorchConfig(BasicMatern52PytorchConfig):
    add_prior: bool = True
    name = "Matern52withPrior"


class BasicMatern32PytorchConfig(BaseElementaryKernelPytorchConfig):
    base_lengthscale: Union[float, Sequence[float]] = BASE_KERNEL_LENGTHSCALE
    base_variance: float = BASE_KERNEL_VARIANCE
    add_prior: bool = False
    lengthscale_prior_parameters: Tuple[float, float] = KERNEL_LENGTHSCALE_GAMMA
    variance_prior_parameters: Tuple[float, float] = KERNEL_VARIANCE_GAMMA
    name = "BasicMatern32"


class Matern32WithPriorPytorchConfig(BasicMatern32PytorchConfig):
    add_prior: bool = True
    name = "Matern32withPrior"


class BasicPeriodicKernelPytorchConfig(BaseElementaryKernelPytorchConfig):
    base_lengthscale: Union[float, Sequence[float]] = BASE_KERNEL_LENGTHSCALE
    base_variance: float = BASE_KERNEL_VARIANCE
    base_period: float = BASE_KERNEL_PERIOD
    add_prior: bool = False
    lengthscale_prior_parameters: Tuple[float, float] = KERNEL_LENGTHSCALE_GAMMA
    variance_prior_parameters: Tuple[float, float] = KERNEL_VARIANCE_GAMMA
    period_prior_parameters: Tuple[float, float] = PERIODIC_KERNEL_PERIOD_GAMMA
    name = "BasicPeriodic"


class PeriodicWithPriorPytorchConfig(BasicPeriodicKernelPytorchConfig):
    add_prior: bool = True
    name = "PeriodicWithPrior"


class BasicRQKernelPytorchConfig(BaseElementaryKernelPytorchConfig):
    base_lengthscale: Union[float, Sequence[float]] = BASE_KERNEL_LENGTHSCALE
    base_variance: float = BASE_KERNEL_VARIANCE
    base_alpha: float = BASE_RQ_KERNEL_ALPHA
    add_prior: bool = False
    lengthscale_prior_parameters: Tuple[float, float] = KERNEL_LENGTHSCALE_GAMMA
    variance_prior_parameters: Tuple[float, float] = KERNEL_VARIANCE_GAMMA
    alpha_prior_parameters: Tuple[float, float] = RQ_KERNEL_ALPHA_GAMMA
    name = "BasicRQ"


class RQWithPriorPytorchConfig(BasicRQKernelPytorchConfig):
    add_prior: bool = True
    name = "RQwithPrior"


class BasicLinearKernelPytorchConfig(BaseElementaryKernelPytorchConfig):
    base_variance: float = BASE_KERNEL_VARIANCE
    base_offset: float = BASE_LINEAR_KERNEL_OFFSET
    add_prior: bool = False
    variance_prior_parameters: Tuple[float, float] = KERNEL_VARIANCE_GAMMA
    offset_prior_parameters: Tuple[float, float] = LINEAR_KERNEL_OFFSET_GAMMA
    name = "BasicLinear"


class LinearWithPriorPytorchConfig(BasicLinearKernelPytorchConfig):
    add_prior: bool = True
    name = "LinearWithPrior"
