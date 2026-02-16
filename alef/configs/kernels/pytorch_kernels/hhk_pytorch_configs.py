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
from alef.configs.kernels.pytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import RBFWithPriorPytorchConfig
from alef.configs.prior_parameters import HHK_SMOOTHING_PRIOR_GAMMA


class BasicHHKPytorchConfig(BaseKernelPytorchConfig):
    base_kernel_config = RBFWithPriorPytorchConfig(input_dimension=0)
    base_smoothing: float = 1.0
    smoothing_prior_parameters: Tuple[float, float] = HHK_SMOOTHING_PRIOR_GAMMA
    add_prior: bool = True
    base_hyperplane_mu: float = 0.0
    base_hyperplane_std: float = 1.0
    topology: int
    name: str = "BasicHHKPytorch"


class HHKEightLocalDefaultPytorchConfig(BasicHHKPytorchConfig):
    topology: int = 3
    name: str = "HHKEightLocalDefaultPytorch"


class HHKFourLocalDefaultPytorchConfig(BasicHHKPytorchConfig):
    topology: int = 2
    name: str = "HHKFourLocalDefaultPytorch"


class HHKTwoLocalDefaultPytorchConfig(BasicHHKPytorchConfig):
    topology: int = 1
    name: str = "HHKTwoLocalDefaultPytorch"


if __name__ == "__main__":
    pass
