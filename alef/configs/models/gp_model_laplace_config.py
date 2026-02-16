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
from alef.configs.kernels.hhk_configs import HHKEightLocalDefaultConfig
from alef.models.gp_model_laplace import PredictionType
from alef.configs.prior_parameters import EXPECTED_OBSERVATION_NOISE


class BasicGPModelLaplaceConfig(BaseModelConfig):
    kernel_config: BaseKernelConfig
    observation_noise: float = 0.01
    expected_observation_noise: float = EXPECTED_OBSERVATION_NOISE
    train_likelihood_variance: bool = True
    pertube_parameters_at_start: bool = True
    perform_multi_start_optimization: bool = True
    prediction_type: PredictionType = PredictionType.NORMAL_APPROXIMATION
    n_starts_for_multistart_opt: int = 5
    pertubation_for_multistart_opt: float = 0.5
    name = "GPModelLaplace"


if __name__ == "__main__":
    kernel_config = HHKEightLocalDefaultConfig(input_dimension=2)
    config = BasicGPModelLaplaceConfig(kernel_config=kernel_config, observation_noise=0.01)
    print(config.dict())
