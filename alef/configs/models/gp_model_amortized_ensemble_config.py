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

from typing import List
from alef.configs.models.base_model_config import BaseModelConfig
from alef.enums.global_model_enums import PredictionQuantity
from alef.models.amortized_infer_structured_kernels.config.nn.amortized_infer_models_configs import (
    BasicAmortizedInferenceModelConfig,
    SmallerStandardNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
    SmallerStandardSmallNoiseBoundNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
)
from alef.models.amortized_infer_structured_kernels.gp.base_symbols import BaseKernelTypes
from alef.utils.gaussian_mixture_density import EntropyApproximation


class BasicGPModelAmortizedEnsembleConfig(BaseModelConfig):
    kernel_list: List[List[BaseKernelTypes]]
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    entropy_approximation: EntropyApproximation = EntropyApproximation.MOMENT_MATCHED_GAUSSIAN
    amortized_model_config: BasicAmortizedInferenceModelConfig
    checkpoint_path: str = ""


class ExperimentalAmortizedEnsembleConfig(BasicGPModelAmortizedEnsembleConfig):
    name: str = "ExperimentalAmortizedEnsemble"
    kernel_list: List[List[BaseKernelTypes]] = [
        [BaseKernelTypes.SE],
        [BaseKernelTypes.SE, BaseKernelTypes.LIN],
        [BaseKernelTypes.SE_MULT_LIN],
        [BaseKernelTypes.SE, BaseKernelTypes.LIN, BaseKernelTypes.SE_MULT_LIN],
        [BaseKernelTypes.SE_MULT_LIN, BaseKernelTypes.LIN],
        [BaseKernelTypes.PER],
        [BaseKernelTypes.PER, BaseKernelTypes.LIN],
        [BaseKernelTypes.LIN_MULT_PER],
        [BaseKernelTypes.PER, BaseKernelTypes.LIN, BaseKernelTypes.LIN_MULT_PER],
        [BaseKernelTypes.LIN_MULT_PER, BaseKernelTypes.LIN],
    ]
    amortized_model_config: BasicAmortizedInferenceModelConfig = (
        SmallerStandardNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig()
    )


class PaperAmortizedEnsembleConfig(BasicGPModelAmortizedEnsembleConfig):
    name: str = "PaperAmortizedEnsemble"
    kernel_list: List[List[BaseKernelTypes]] = [
        [BaseKernelTypes.SE],
        [BaseKernelTypes.SE, BaseKernelTypes.LIN],
        [BaseKernelTypes.SE_MULT_LIN],
        [BaseKernelTypes.SE, BaseKernelTypes.LIN, BaseKernelTypes.SE_MULT_LIN],
        [BaseKernelTypes.SE_MULT_LIN, BaseKernelTypes.LIN],
        [BaseKernelTypes.PER],
        [BaseKernelTypes.PER, BaseKernelTypes.LIN],
        [BaseKernelTypes.LIN_MULT_PER],
        [BaseKernelTypes.PER, BaseKernelTypes.LIN, BaseKernelTypes.LIN_MULT_PER],
        [BaseKernelTypes.LIN_MULT_PER, BaseKernelTypes.LIN],
    ]  # this list is only the default list - not used in the paper - it can be changed after initialization
    amortized_model_config: BasicAmortizedInferenceModelConfig = (
        SmallerStandardNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig()
    )


class AmortizedEnsembleWithMaternConfig(BasicGPModelAmortizedEnsembleConfig):
    name: str = "AmortizedEnsembleWithMatern"
    kernel_list: List[List[BaseKernelTypes]] = [
        [BaseKernelTypes.SE],
        [BaseKernelTypes.SE, BaseKernelTypes.LIN],
        [BaseKernelTypes.SE_MULT_LIN],
        [BaseKernelTypes.SE, BaseKernelTypes.LIN, BaseKernelTypes.SE_MULT_LIN],
        [BaseKernelTypes.SE_MULT_LIN, BaseKernelTypes.LIN],
        [BaseKernelTypes.MATERN52],
        [BaseKernelTypes.MATERN52, BaseKernelTypes.LIN],
        [BaseKernelTypes.LIN_MULT_MATERN52],
        [BaseKernelTypes.MATERN52, BaseKernelTypes.LIN, BaseKernelTypes.LIN_MULT_MATERN52],
        [BaseKernelTypes.LIN_MULT_MATERN52, BaseKernelTypes.LIN],
        [BaseKernelTypes.PER],
        [BaseKernelTypes.PER, BaseKernelTypes.LIN],
        [BaseKernelTypes.LIN_MULT_PER],
    ]  # this list is only the default list - not used in the paper - it can be changed after initialization
    amortized_model_config: BasicAmortizedInferenceModelConfig = (
        SmallerStandardSmallNoiseBoundNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig()
    )
