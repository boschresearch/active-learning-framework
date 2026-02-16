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
from alef.configs.prior_parameters import NOISE_VARIANCE_EXPONENTIAL_LAMBDA


class BasicGPModelPytorchConfig(BaseModelConfig):
    kernel_config: BaseKernelPytorchConfig
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    initial_likelihood_noise: float = 0.2
    set_prior_on_observation_noise: bool = False
    observation_noise_variance_lambda: float = NOISE_VARIANCE_EXPONENTIAL_LAMBDA
    fix_likelihood_variance: bool = False
    add_constant_mean_function: bool = False
    optimize_hps: bool = True
    training_iter: int = 150
    lr: float = 0.1
    do_multi_start_optimization: bool = False
    n_restarts_multistart: int = 10
    do_map_estimation: bool = False
    do_early_stopping: bool = True
    name = "GPModelPytorch"


class GPModelPytorchMAPConfig(BasicGPModelPytorchConfig):
    do_map_estimation: bool = True
    set_prior_on_observation_noise: bool = True
    name = "GPModelPytorchMAP"


class GPModelPytorchMAPIntenseOptimizationConfig(BasicGPModelPytorchConfig):  # Model class tilted to be used with HHK
    do_map_estimation: bool = True
    do_early_stopping: bool = True
    set_prior_on_observation_noise: bool = True
    do_multi_start_optimization: int = 20
    training_iter: int = 500
    lr: float = 0.1
    name = "GPModelPytorchMAPIntenseOptimizationNoNoisePrior"


class GPModelPytorchMultistartConfig(BasicGPModelPytorchConfig):
    do_multi_start_optimization: bool = True
    set_prior_on_observation_noise: bool = True  # in order to sample for multistart
    name = "GPModelPytorchMultistart"
