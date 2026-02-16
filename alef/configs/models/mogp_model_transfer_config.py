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
from alef.enums.global_model_enums import PredictionQuantity, InitialParameters
from alef.models.mogp_model_transfer import SourceTraining


class BasicTransferGPModelConfig(BaseModelConfig):
    kernel_config: BaseKernelConfig
    observation_noise: float = 0.1
    expected_observation_noise: float = 0.3
    optimize_hps: bool = True
    train_likelihood_variance: bool = True
    source_training_mode: SourceTraining = SourceTraining.WITHOUT_TARGET
    sample_initial_parameters_at_start: bool = True
    initial_parameter_strategy: InitialParameters = InitialParameters.PERTURB
    perturbation_for_multistart_opt: float = 0.5
    perturbation_for_singlestart_opt: float = 0.1
    perform_multi_start_optimization: bool = True
    n_starts_for_multistart_opt: int = 5
    set_prior_on_observation_noise: bool = False
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    name: str = "BasicTransferGP"


if __name__ == "__main__":
    pass
