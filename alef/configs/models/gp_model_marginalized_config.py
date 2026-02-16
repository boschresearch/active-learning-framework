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
from alef.models.gp_model_marginalized import InitializationType, PredictionQuantity
from alef.configs.prior_parameters import EXPECTED_OBSERVATION_NOISE
from alef.utils.gaussian_mixture_density import EntropyApproximation


class BasicGPModelMarginalizedConfig(BaseModelConfig):
    kernel_config: BaseKernelConfig
    observation_noise: float = 0.01
    expected_observation_noise: float = EXPECTED_OBSERVATION_NOISE
    train_likelihood_variance: bool = True
    num_samples: int = 100
    num_burnin_steps: int = 500
    thin_trace: bool = True
    thin_steps: int = 50
    initialization_type: InitializationType = InitializationType.PRIOR_DRAW
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    entropy_approximation: EntropyApproximation = EntropyApproximation.MOMENT_MATCHED_GAUSSIAN
    name = "GPModelMarginalized"


class GPModelMarginalizedConfigMoreThinningConfig(BasicGPModelMarginalizedConfig):
    thin_steps: int = 100
    name = "GPModelMarginalizedMoreThinning"


class GPModelMarginalizedConfigMoreSamplesConfig(BasicGPModelMarginalizedConfig):
    num_samples: int = 150
    name = "GPModelMarginalizedMoreSamples"


class GPModelMarginalizedConfigMoreSamplesMoreThinningConfig(BasicGPModelMarginalizedConfig):
    num_burnin_steps: int = 1500
    thin_steps: int = 100
    name = "GPModelMarginalizedMoreSamplesMoreThinning"


class GPModelMarginalizedConfigMAPInitialized(BasicGPModelMarginalizedConfig):
    initialization_type: InitializationType = InitializationType.MAP_ESTIMATE
    name = "GPModelMarginalizedMAPInitialized"


class GPModelMarginalizedConfigFast(BasicGPModelMarginalizedConfig):
    initialization_type: InitializationType = InitializationType.MAP_ESTIMATE
    num_samples: int = 50
    num_burnin_steps: int = 100
    thin_steps: int = 20
    name = "GPModelMarginalizedFast"


if __name__ == "__main__":
    kernel_config = HHKEightLocalDefaultConfig(input_dimension=2)
    config = GPModelMarginalizedConfigMAPInitialized(kernel_config=kernel_config, observation_noise=0.01)
    print(config.dict())
