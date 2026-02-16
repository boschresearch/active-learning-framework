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

from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import RBFWithPriorPytorchConfig
from alef.configs.models.base_model_config import BaseModelConfig
from alef.configs.models.gp_model_pytorch_config import BasicGPModelPytorchConfig


class BasicGPModelPytorchEnsembleConfig(BaseModelConfig):
    name: str = "BaseGPModelPytorchEnsembleConfig"
    gp_model_config: BasicGPModelPytorchConfig = BasicGPModelPytorchConfig(
        kernel_config=RBFWithPriorPytorchConfig(input_dimension=1)
    )
