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

from enum import Enum


class AcquisitionFunctionType(Enum):
    GP_UCB = 1
    EXPECTED_IMPROVEMENT = 2
    RANDOM = 3
    EXPECTED_IMPROVEMENT_PER_SECOND = 4
    INTEGRATED_EXPECTED_IMPROVEMENT = 5
    SAFE_OPT = 1001
    SAFE_GP_UCB = 1002
    SAFE_EXPECTED_IMPROVEMENT = 1003


class ValidationType(Enum):
    SIMPLE_REGRET = 1
    CUMMULATIVE_REGRET = 2
    MAX_OBSERVED = 3

    @staticmethod
    def get_name(val_type):
        if val_type == ValidationType.SIMPLE_REGRET:
            return "SIMPLEREGERT"
        elif val_type == ValidationType.CUMMULATIVE_REGRET:
            return "CUMMULATIVEREGRET"
        elif val_type == ValidationType.MAX_OBSERVED:
            return "MAXOBSERVED"


class AcquisitionOptimizationType(Enum):
    RANDOM_SHOOTING = 1
    EVOLUTIONARY = 2


class AcquisitionOptimizationObjectBOType(Enum):
    TRAILING_CANDIDATES = 1
    EVOLUTIONARY = 2


if __name__ == "__main__":
    print(ValidationType.get_name(ValidationType.INSTANTANEOUS_REGRET))
