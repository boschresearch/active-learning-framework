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

from typing import Union, List, Optional
from alef.configs.acquisition_function.al_acquisition_functions.acq_random_config import BasicRandomConfig
from alef.configs.acquisition_function.al_acquisition_functions.base_batch_al_acquisition_function_config import (
    BaseBatchALAcquisitionFunctionConfig,
)
from alef.configs.acquisition_function.al_acquisition_functions.pred_entropy_batch_config import (
    BasicPredEntropyBatchConfig,
)
from alef.enums.active_learner_enums import (
    ValidationType,
)
from pydantic import BaseSettings


class BasicBatchActiveLearnerConfig(BaseSettings):
    acquisition_function_config: Union[BaseBatchALAcquisitionFunctionConfig, BasicRandomConfig]
    validation_type: ValidationType = ValidationType.RMSE
    validation_at: Optional[List[int]] = None
    batch_size: int = 5
    use_smaller_acquistion_set: bool = True
    smaller_set_size: int = 200


class EntropyBatchActiveLearnerConfig(BasicBatchActiveLearnerConfig):
    acquisition_function_config: Union[BaseBatchALAcquisitionFunctionConfig, BasicRandomConfig] = (
        BasicPredEntropyBatchConfig()
    )


class RandomBatchActiveLearnerConfig(BasicBatchActiveLearnerConfig):
    acquisition_function_config: BasicRandomConfig = BasicRandomConfig()


if __name__ == "__main__":
    config = RandomBatchActiveLearnerConfig()
