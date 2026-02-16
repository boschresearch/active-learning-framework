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

from typing import Union
from alef.acquisition_function.acquisition_function_factory import AcquisitionFunctionFactory
from alef.active_learner.active_learner import ActiveLearner
from alef.active_learner.active_learner_oracle import ActiveLearnerOracle
from alef.active_learner.continuous_policy_active_learner import ContinuousPolicyActiveLearner
from alef.active_learner.batch_active_learner import BatchActiveLearner

from alef.configs.active_learner.active_learner_configs import BasicActiveLearnerConfig
from alef.configs.active_learner.active_learner_oracle_configs import BasicActiveLearnerOracleConfig
from alef.configs.active_learner.continuous_policy_active_learner_configs import (
    BasicContinuousPolicyActiveLearnerOracleConfig,
)
from alef.configs.active_learner.batch_active_learner_configs import BasicBatchActiveLearnerConfig


class ActiveLearnerFactory:
    @staticmethod
    def build(
        active_learner_config: Union[
            BasicActiveLearnerConfig, BasicActiveLearnerOracleConfig, BasicBatchActiveLearnerConfig
        ],
    ):
        if isinstance(active_learner_config, BasicActiveLearnerConfig):
            acquisition_function_config = active_learner_config.acquisition_function_config
            acquisition_function = AcquisitionFunctionFactory.build(acquisition_function_config)
            return ActiveLearner(acquisition_function=acquisition_function, **active_learner_config.dict())
        elif isinstance(active_learner_config, BasicActiveLearnerOracleConfig):
            acquisition_function_config = active_learner_config.acquisition_function_config
            acquisition_function = AcquisitionFunctionFactory.build(acquisition_function_config)
            return ActiveLearnerOracle(acquisition_function=acquisition_function, **active_learner_config.dict())
        elif isinstance(active_learner_config, BasicContinuousPolicyActiveLearnerOracleConfig):
            return ContinuousPolicyActiveLearner(**active_learner_config.dict())
        elif isinstance(active_learner_config, BasicBatchActiveLearnerConfig):
            acquisition_function_config = active_learner_config.acquisition_function_config
            acquisition_function = AcquisitionFunctionFactory.build(acquisition_function_config)
            return BatchActiveLearner(acquisition_function=acquisition_function, **active_learner_config.dict())
        else:
            raise NotImplementedError("Invalid config")
