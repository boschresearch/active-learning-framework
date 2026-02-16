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
from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.enums.global_model_enums import PredictionQuantity


class BasicMetaGPModelConfig(BaseModelConfig):
    zero_mean: bool = False
    kernel_config: BaseKernelConfig
    observation_noise: float = 0.1
    min_calib_freq: float = 0.95
    weight_decay: float = 1e-4
    num_iter_fit: int = 5000
    predict_outputscale: bool = True
    train_data_in_kl: bool = True
    optimize_hps: bool = True
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    name: str = "BasicMetaGP"
    perform_multi_start_optimization: bool = True
    train_likelihood_variance: bool = True


if __name__ == "__main__":
    pass
