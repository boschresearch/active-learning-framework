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

from typing import Union, Sequence
from alef.acquisition_function.al_acquisition_functions.base_al_acquisition_function import BaseALAcquisitionFunction
from alef.acquisition_function.bo_acquisition_functions.base_bo_acquisition_function import BaseBOAcquisitionFunction
from alef.configs.acquisition_function.safe_acquisition_functions.base_safe_acquisition_function_config import (
    BaseSafeAcquisitionFunctionConfig,
)
from alef.acquisition_function.al_acquisition_functions.pred_variance import PredVariance
from alef.acquisition_function.bo_acquisition_functions.ei import EI


class BasicSafeDiscoverConfig(BaseSafeAcquisitionFunctionConfig):
    safety_thresholds_lower: Union[float, Sequence[float]]
    safety_thresholds_upper: Union[float, Sequence[float]]
    alpha: float = 0.02275
    acquisition_function: Union[BaseALAcquisitionFunction, BaseBOAcquisitionFunction] = PredVariance()
    name: str = "safe_discover"


class BasicSafeDiscoverQuantileConfig(BaseSafeAcquisitionFunctionConfig):
    safety_thresholds_lower: Union[float, Sequence[float]]
    safety_thresholds_upper: Union[float, Sequence[float]]
    alpha: float = 0.02275
    acquisition_function: Union[BaseALAcquisitionFunction, BaseBOAcquisitionFunction] = PredVariance()
    score_threshold_quantile: float = 0.5
    name: str = "safe_discover_quantile"


class BasicSafeDiscoverEIConfig(BasicSafeDiscoverConfig):
    acquisition_function: Union[BaseALAcquisitionFunction, BaseBOAcquisitionFunction] = EI(xi=0.01)
    name: str = "safe_discover_ei"


class BasicSafeDiscoverQuantileEIConfig(BasicSafeDiscoverQuantileConfig):
    acquisition_function: Union[BaseALAcquisitionFunction, BaseBOAcquisitionFunction] = EI(xi=0.01)
    name: str = "safe_discover_quantile_ei"
