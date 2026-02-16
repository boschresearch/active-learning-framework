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

from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.configs.models.feature_extractors.base_feature_extractor_config import BaseFeatureExtractorConfig
from typing import Tuple


class BaseDeepKernelConfig(BaseKernelConfig):
    add_prior: bool
    base_lengthscale: float = 1.0
    base_variance: float = 1.0
    lengthscale_trainable: bool = True
    lengthscale_prior_parameters: Tuple[float, float] = (2.0, 2.0)
    variance_prior_parameters: Tuple[float, float] = (2.0, 3.0)
    feature_extractor_config: BaseFeatureExtractorConfig
    name: str = "BaseDeepKernel"
