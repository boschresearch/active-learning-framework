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


class BasicNKNConfig(BaseKernelConfig):
    middle_layer_list: List[int] = [4, 2]
    base_lengthscale: float = 1.0
    base_variance: float = 1.0
    base_period: float = 1.0
    base_alpha: float = 1.0
    init_linear_layer_a: float = 0.0
    init_linear_layer_b: float = 1.0
    name: str = "BasicNKN"
