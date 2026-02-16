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


class BaseMLPConfig(BaseFeatureExtractorConfig):
    layer_size_list: List[int] = [50, 20, 2]
    prior_w_sigma: float = 0.2
    prior_b_sigma: float = 0.05
    add_prior: bool = False
    name: str = "BaseMLP"


class MLPWithPriorConfig(BaseMLPConfig):
    add_prior: bool = True
    name = "MLPWithPrior"


class SmallMLPWithPriorConfig(MLPWithPriorConfig):
    layer_size_list: List[int] = [15, 10, 2]
    name = "SmallMLPWithPrior"
