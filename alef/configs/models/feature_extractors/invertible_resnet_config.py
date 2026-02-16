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
from alef.configs.models.feature_extractors.base_feature_extractor_config import BaseFeatureExtractorConfig


class InvertibleResnetConfig(BaseFeatureExtractorConfig):
    num_layers: int = 20
    residual_layer_size_list: List[int] = [20]
    share_weights: bool = True
    W_scale: float = 0.1
    b_scale: float = 0.1
    add_prior: bool = False
    add_regularization: bool = False
    regularizer_lambdas: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    contrain_regularizer_n_data: bool = False
    contrained_regularizer_k: int = 50
    add_noise_on_train_mode: bool = False
    layer_noise_std: float = 0.1
    name: str = "InvertibleResnet"


class InvertibleResnetWithPriorConfig(InvertibleResnetConfig):
    add_prior: bool = True
    name: str = "InvertibleResnetWithPrior"


class InvertibleResnetWithLayerNoiseConfig(InvertibleResnetWithPriorConfig):
    add_noise_on_train_mode: bool = True
    layer_noise_std: float = 0.1
    name: str = "InvertibleResnetWithLayerNoise"


class CurlRegularizedIResnetConfig(InvertibleResnetWithPriorConfig):
    add_regularization: bool = True
    contrain_regularizer_n_data: bool = True
    contrained_regularizer_k: int = 50
    regularizer_lambdas: List[float] = [0.0, 0.0, 0.0, 0.0, 10.0, 0.0]
    name: str = "CurlRegularizedInvertibleResnet"


class AxisRegularizedIResnetConfig(InvertibleResnetWithPriorConfig):
    add_regularization: bool = True
    contrain_regularizer_n_data: bool = True
    contrained_regularizer_k: int = 50
    regularizer_lambdas: List[float] = [0.0, 0.0, 0.0, 10.0, 0.0, 0.0]
    name: str = "AxisRegularizedInvertibleResnet"


class ExploreRegularizerIResnetConfig(InvertibleResnetWithPriorConfig):
    add_regularization: bool = True
    contrain_regularizer_n_data: bool = False
    add_prior: bool = True
    contrained_regularizer_k: int = 50
    regularizer_lambdas: List[float] = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    name: str = "ExploreRegularizerIResnet"
