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
from alef.enums.global_model_enums import InitialParameters, PredictionQuantity
from alef.configs.prior_parameters import EXPECTED_OBSERVATION_NOISE


class BasicSparseGPModelConfig(BaseModelConfig):
    kernel_config: BaseKernelConfig
    observation_noise: float = 0.01
    optimize_hps: bool = True
    n_inducing_points: int = 100
    train_likelihood_variance: bool = True
    sample_initial_parameters_at_start: bool = True
    initial_parameter_strategy: InitialParameters = InitialParameters.UNIFORM_DISTRIBUTION
    perturbation_for_multistart_opt: float = 0.5
    perturbation_for_singlestart_opt: float = 0.1
    initial_uniform_lower_bound: float = -1.0
    initial_uniform_upper_bound: float = 1.0
    perform_multi_start_optimization: bool = True
    n_starts_for_multistart_opt: int = 4
    set_prior_on_observation_noise: bool = False
    expected_observation_noise: float = EXPECTED_OBSERVATION_NOISE
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    name = "SparseGPModel"


class SparseGPModel300IPConfig(BasicSparseGPModelConfig):
    n_inducing_points: int = 300
    name = "SparseGPModel300IP"


class SparseGPModel500IPConfig(BasicSparseGPModelConfig):
    n_inducing_points: int = 500
    name = "SparseGPModel500IP"


class SparseGPModel700IPConfig(BasicSparseGPModelConfig):
    n_inducing_points: int = 700
    name = "SparseGPModel700IP"


class SparseGPModel700IPExtenseConfig(BasicSparseGPModelConfig):
    n_inducing_points: int = 700
    n_starts_for_multistart_opt: int = 20
    name = "SparseGPModel700IPExtense"


class SparseGPModelFastConfig(BasicSparseGPModelConfig):
    perform_multi_start_optimization: bool = False
    name = "SparseGPModelFast"


class SparseGPModelFixedNoiseConfig(BasicSparseGPModelConfig):
    train_likelihood_variance: bool = False
    name = "SparseGPModelFixedNoise"


if __name__ == "__main__":
    kernel_config = HHKEightLocalDefaultConfig(input_dimension=2)
    config = BasicSparseGPModelConfig(kernel_config=kernel_config)
    config.observation_noise = 0.02
    print(config.dict())
