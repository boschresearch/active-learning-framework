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

from typing import List, Optional
from alef.enums.active_learner_enums import ValidationType
from pydantic import BaseSettings


class BasicContinuousPolicyActiveLearnerOracleConfig(BaseSettings):
    validation_type: ValidationType = ValidationType.RMSE
    validation_at: Optional[List[int]] = None
    policy_path: str = ""


class PytestContinuousPolicyActiveLearnerOracleConfig(BasicContinuousPolicyActiveLearnerOracleConfig):
    pytest: bool = True
    policy_dimension: int


if __name__ == "__main__":
    pass
