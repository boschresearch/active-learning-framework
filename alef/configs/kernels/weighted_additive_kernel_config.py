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

from typing import List, Tuple
from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from alef.configs.kernels.matern52_configs import Matern52WithPriorConfig, BasicMatern52Config
from alef.configs.kernels.linear_configs import LinearWithPriorConfig, BasicLinearConfig
from alef.configs.base_parameters import BASE_KERNEL_VARIANCE
from alef.configs.prior_parameters import KERNEL_VARIANCE_GAMMA, WEIGHTED_ADDITIVE_KERNEL_ALPHA_NORMAL


class BasicWeightedAdditiveKernelConfig(BaseKernelConfig):
    base_kernel_config_list: List[BaseKernelConfig] = [
        BasicRBFConfig(input_dimension=0),
        BasicMatern52Config(input_dimension=0),
        BasicLinearConfig(input_dimension=0),
    ]
    add_prior: bool = False
    alpha_prior_parameters: Tuple[float, float] = WEIGHTED_ADDITIVE_KERNEL_ALPHA_NORMAL
    base_variance: float = BASE_KERNEL_VARIANCE
    use_own_variance: bool = True
    variance_prior_parameters: Tuple[float, float] = KERNEL_VARIANCE_GAMMA
    name: str = "BasicWeightedAdditiveKernel"


class WeightedAdditiveKernelWithPriorConfig(BasicWeightedAdditiveKernelConfig):
    base_kernel_config_list: List[BaseKernelConfig] = [
        RBFWithPriorConfig(input_dimension=0),
        Matern52WithPriorConfig(input_dimension=0),
        LinearWithPriorConfig(input_dimension=0),
    ]
    add_prior: bool = True
    name: str = "WeightedAdditiveKernelWithPrior"
