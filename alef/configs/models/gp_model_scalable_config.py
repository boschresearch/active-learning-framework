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
from alef.models.gp_model_scalable import BaseGPType, OptimizerType, ValidationMetric
from alef.configs.prior_parameters import EXPECTED_OBSERVATION_NOISE


class BasicScalableGPModelConfig(BaseModelConfig):
    kernel_config: BaseKernelConfig
    initial_observation_noise: float = 0.01
    base_gp_type: BaseGPType = BaseGPType.SGPR
    optimizer_type: OptimizerType = OptimizerType.ADAM
    n_inducing_points: int = 300
    n_iterations: int = 3000
    learning_rate: float = 0.01
    use_mini_batches: bool = False
    use_validation_set: bool = False
    set_prior_on_observation_noise: bool = False
    expected_observation_noise: float = EXPECTED_OBSERVATION_NOISE
    val_fraction: float = 0.1
    validation_metric: ValidationMetric = ValidationMetric.RMSE
    n_repeats = 5
    name = "GPModelScalable"


class GPRAdamConfig(BasicScalableGPModelConfig):
    base_gp_type: BaseGPType = BaseGPType.GPR
    learning_rate: float = 0.002
    name = "GPRWithAdam"


class GPRAdamWithValidationSet(BasicScalableGPModelConfig):
    base_gp_type: BaseGPType = BaseGPType.GPR
    use_validation_set: bool = True


class GPRAdamWithValidationSetNLL(BasicScalableGPModelConfig):
    base_gp_type: BaseGPType = BaseGPType.GPR
    validation_metric: ValidationMetric = ValidationMetric.NLL
    use_validation_set: bool = True


if __name__ == "__main__":
    pass
