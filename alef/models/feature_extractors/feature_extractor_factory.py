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

from alef.configs.models.feature_extractors.base_feature_extractor_config import BaseFeatureExtractorConfig
from alef.configs.models.feature_extractors.invertible_resnet_config import InvertibleResnetConfig
from alef.configs.models.feature_extractors.mlp_configs import BaseMLPConfig
from alef.models.feature_extractors.invertible_resnet import InvertibleResNet
from alef.models.feature_extractors.multi_layer_perceptron import MultiLayerPerceptron


class FeatureExtractorFactory:
    @staticmethod
    def build(extractor_config: BaseFeatureExtractorConfig, input_dimension: int):
        if isinstance(extractor_config, InvertibleResnetConfig):
            feature_extractor = InvertibleResNet(input_dimension=input_dimension, **extractor_config.dict())
            return feature_extractor
        elif isinstance(extractor_config, BaseMLPConfig):
            feature_extractor = MultiLayerPerceptron(input_dimension=input_dimension, **extractor_config.dict())
            return feature_extractor
