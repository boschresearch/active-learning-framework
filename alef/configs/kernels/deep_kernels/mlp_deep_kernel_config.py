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

from alef.configs.kernels.deep_kernels.base_deep_kernel_config import BaseDeepKernelConfig
from alef.configs.models.feature_extractors.base_feature_extractor_config import BaseFeatureExtractorConfig
from alef.configs.models.feature_extractors.mlp_configs import (
    BaseMLPConfig,
    MLPWithPriorConfig,
    SmallMLPWithPriorConfig,
)


class BasicMLPDeepKernelConfig(BaseDeepKernelConfig):
    add_prior: bool = False
    feature_extractor_config: BaseFeatureExtractorConfig = BaseMLPConfig()
    name = "BasicMLPDeepKernel"


class MLPWithPriorDeepKernelConfig(BaseDeepKernelConfig):
    add_prior: bool = True
    feature_extractor_config: BaseFeatureExtractorConfig = MLPWithPriorConfig()
    name = "MLPWithPriorDeepKernel"


class SmallMLPWithPriorDeepKernelConfig(BaseDeepKernelConfig):
    add_prior: bool = True
    feature_extractor_config: BaseFeatureExtractorConfig = SmallMLPWithPriorConfig()
    name = "SmallMLPWithPriorDeepKernel"
