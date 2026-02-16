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

from alef.configs.models.base_model_config import BaseModelConfig
from alef.enums.global_model_enums import PredictionQuantity
from alef.configs.kernels.pytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig


class BasicSVGPModelPytorchConfig(BaseModelConfig):
    kernel_config: BaseKernelPytorchConfig
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    initial_likelihood_noise: float = 0.2
    fix_likelihood_variance: bool = False
    add_constant_mean_function: bool = False
    n_epochs: int = 500
    batch_size: int = 64
    lr: float = 0.03
    n_inducing_points: int = 100
    use_fraction_for_inducing_points: bool = True
    fraction_inducing_points: float = 0.1
    batch_size_is_dataset_size: bool = True
    name = "SVGPModelPytorch"
