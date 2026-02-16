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
from alef.enums.global_model_enums import InitializationType, PredictionQuantity


class BasicSOMOGPModelMarginalizedConfig(BaseModelConfig):
    kernel_config: BaseKernelConfig
    observation_noise: float = 0.1
    expected_observation_noise: float = 0.3
    train_likelihood_variance: bool = True
    num_samples: int = 100
    num_burnin_steps: int = 500
    thin_trace: bool = True
    thin_steps: int = 50
    initialization_type: InitializationType = InitializationType.PRIOR_DRAW
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    name = "BasicSOMOGPMarginalized"


class SOMOGPModelMarginalizedConfigMoreThinningConfig(BasicSOMOGPModelMarginalizedConfig):
    thin_steps: int = 100
    name = "BasicSOMOGPMarginalizedMoreThinning"


class SOMOGPModelMarginalizedConfigMoreSamplesConfig(BasicSOMOGPModelMarginalizedConfig):
    num_samples: int = 150
    name = "BasicSOMOGPMarginalizedMoreSamples"


class SOMOGPModelMarginalizedConfigMoreSamplesMoreThinningConfig(BasicSOMOGPModelMarginalizedConfig):
    num_burnin_steps: int = 1500
    thin_steps: int = 100
    name = "BasicSOMOGPMarginalizedMoreSamplesMoreThinning"


class SOMOGPModelMarginalizedConfigMAPInitialized(BasicSOMOGPModelMarginalizedConfig):
    initialization_type: InitializationType = InitializationType.MAP_ESTIMATE
    name = "BasicSOMOGPMarginalizedMAPInitialized"


class SOMOGPModelMarginalizedConfigFast(BasicSOMOGPModelMarginalizedConfig):
    initialization_type: InitializationType = InitializationType.MAP_ESTIMATE
    num_samples: int = 50
    num_burnin_steps: int = 100
    thin_steps: int = 10
    name = "BasicSOMOGPMarginalizedFast"
