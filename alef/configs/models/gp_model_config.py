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
from alef.enums.global_model_enums import PredictionQuantity, InitialParameters
from alef.configs.prior_parameters import EXPECTED_OBSERVATION_NOISE


class BasicGPModelConfig(BaseModelConfig):
    kernel_config: BaseKernelConfig
    observation_noise: float = 0.01
    optimize_hps: bool = True
    train_likelihood_variance: bool = True
    sample_initial_parameters_at_start: bool = True
    initial_parameter_strategy: InitialParameters = InitialParameters.PERTURB
    perturbation_for_multistart_opt: float = 0.5
    perturbation_for_singlestart_opt: float = 0.1
    perform_multi_start_optimization: bool = True
    initial_uniform_lower_bound: float = -0.5
    initial_uniform_upper_bound: float = 0.5
    n_starts_for_multistart_opt: int = 10
    set_prior_on_observation_noise: bool = False
    expected_observation_noise: float = EXPECTED_OBSERVATION_NOISE
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    name = "GPModel"


class GPModelWithNoisePriorConfig(BasicGPModelConfig):
    set_prior_on_observation_noise: bool = True
    name = "GPModelWithNoisePrior"


class GPModelExtenseOptimization(BasicGPModelConfig):
    n_starts_for_multistart_opt: int = 20
    name = "GPModelExtenseOptimization"


class GPModelUniformInitializationConfig(BasicGPModelConfig):
    initial_parameter_strategy: InitialParameters = InitialParameters.UNIFORM_DISTRIBUTION
    name = "GPModelUniformInitialization"


class GPModelFastConfig(BasicGPModelConfig):
    initial_parameter_strategy: InitialParameters = InitialParameters.PERTURB
    perform_multi_start_optimization: bool = False
    name = "GPModelFast"


class GPModelFixedNoiseConfig(BasicGPModelConfig):
    train_likelihood_variance: bool = False
    name = "GPModelFixedNoise"


class GPModelSmallPertubationConfig(BasicGPModelConfig):
    initial_parameter_strategy: InitialParameters = InitialParameters.PERTURB
    perturbation_for_multistart_opt: float = 0.1
    name = "GPModelSmallPertubation"


if __name__ == "__main__":
    kernel_config = HHKEightLocalDefaultConfig(input_dimension=2)

    print(GPModelFixedNoiseConfig.__fields__)
    print("kernel_config" in GPModelFixedNoiseConfig.__fields__)
    # config = BasicGPModelConfig(kernel_config=kernel_config)
    # config.observation_noise = 0.02
    # print(config.dict())
