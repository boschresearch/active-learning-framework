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

from alef.configs.kernels.kernel_list_configs import (
    BasicKernelListConfig,
    SEKernelViaKernelListConfig,
    MaternKernelViaKernelListConfig,
)
from alef.configs.models.base_model_config import BaseModelConfig
from alef.enums.global_model_enums import PredictionQuantity
from alef.models.amortized_infer_structured_kernels.config.nn.amortized_infer_models_configs import (
    BasicAmortizedInferenceModelConfig,
    SmallerStandardNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
    SmallerStandardSmallNoiseBoundNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
    ARDRBFOnlyAmortizedModelConfig,
)


class BasicGPModelAmortizedStructuredConfig(BaseModelConfig):
    kernel_config: BasicKernelListConfig
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    amortized_model_config: BasicAmortizedInferenceModelConfig
    checkpoint_path: str = ""
    do_warm_start: bool = False
    warm_start_steps: int = 10
    warm_start_lr: float = 0.1


class PaperAmortizedStructuredConfig(BasicGPModelAmortizedStructuredConfig):
    name: str = "PaperAmortizedStructured"
    kernel_config: BasicKernelListConfig = SEKernelViaKernelListConfig(
        input_dimension=0
    )  # default initialized with SE kernel as input - can be changed after model building
    amortized_model_config: BasicAmortizedInferenceModelConfig = (
        SmallerStandardNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig()
    )


class AmortizedStructuredWithMaternConfig(BasicGPModelAmortizedStructuredConfig):
    name: str = "AmortizedStructuredWithMatern"
    kernel_config: BasicKernelListConfig = MaternKernelViaKernelListConfig(input_dimension=0)
    amortized_model_config: BasicAmortizedInferenceModelConfig = (
        SmallerStandardSmallNoiseBoundNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig()
    )


class ExperimentalAmortizedStructuredConfig(BasicGPModelAmortizedStructuredConfig):
    name: str = "ExperimentalAmortizedStructured"
    amortized_model_config: BasicAmortizedInferenceModelConfig = (
        SmallerStandardNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig()
    )


class AHGGPRBFAmortizedConfig(BasicGPModelAmortizedStructuredConfig):
    name: str = "AHGGPRBFAmortized"
    kernel_config: BasicKernelListConfig = SEKernelViaKernelListConfig(input_dimension=0)
    amortized_model_config: BasicAmortizedInferenceModelConfig = ARDRBFOnlyAmortizedModelConfig()
