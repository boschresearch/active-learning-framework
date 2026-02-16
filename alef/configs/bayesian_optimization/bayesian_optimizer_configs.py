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

from pydantic import BaseSettings
from alef.configs.acquisition_function.bo_acquisition_functions.base_bo_acquisition_function_config import (
    BaseBOAcquisitionFunctionConfig,
)
from alef.configs.acquisition_function.bo_acquisition_functions.ei_config import BasicEIConfig
from alef.configs.acquisition_function.bo_acquisition_functions.gp_ucb_config import BasicGPUCBConfig
from alef.configs.acquisition_function.bo_acquisition_functions.integrated_ei_config import BasicIntegratedEIConfig
from alef.enums.bayesian_optimization_enums import (
    AcquisitionOptimizationType,
    ValidationType,
)


class BaseBOConfig(BaseSettings):
    acquisition_function_config: BaseBOAcquisitionFunctionConfig
    validation_type: ValidationType = ValidationType.MAX_OBSERVED
    acquisiton_optimization_type: AcquisitionOptimizationType = AcquisitionOptimizationType.EVOLUTIONARY
    do_plotting: bool = False
    random_shooting_n: int = 500
    steps_evoluationary: int = 5
    name = "BaseBO"


class BOExpectedImprovementConfig(BaseBOConfig):
    acquisition_function_config: BaseBOAcquisitionFunctionConfig = BasicEIConfig()
    name = "BOExpectedImprovement"


class BOGPUCBConfig(BaseBOConfig):
    acquisition_function_config: BaseBOAcquisitionFunctionConfig = BasicGPUCBConfig()
    name = "BOGPUCB"


class BOIntegratedExpectedImprovementConfig(BaseBOConfig):
    acquisition_function_config: BaseBOAcquisitionFunctionConfig = BasicIntegratedEIConfig()
    name = "BOIntegratedExpectedImprovement"
