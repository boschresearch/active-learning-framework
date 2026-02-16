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
    PRED_VAR = 1
    PRED_VAR_SAFETY = 2
    PRED_VAR_COMBINED = 3
    HYPER_BALD = 4
    RANDOM = 5
    PRED_ENTROPY = 6
    PRED_ENTROPY_ALL = 7


class BatchAcquisitionFunctionType(Enum):
    BATCH_BALD = 1
    HYPER_BATCH_BALD = 2
    RANDOM = 3
    ENTROPY = 4


class BatchAcquisitionOptimizationType(Enum):
    GREEDY = 1


class ValidationType(Enum):
    NEG_LOG_LIKELI = 1
    RMSE = 2
    RMSE_MULTIOUTPUT = 3

    @staticmethod
    def get_name(val_type):
        if val_type == ValidationType.NEG_LOG_LIKELI:
            return "NEGLOGLIKELI"
        elif val_type == ValidationType.RMSE:
            return "RMSE"
        elif val_type == ValidationType.RMSE_MULTIOUTPUT:
            return "RMSEMULTIOUTPUT"


class OracleALAcquisitionOptimizationType(Enum):
    RANDOM_SHOOTING = 1


class ModelSelectionType(Enum):
    CROSS_VALIDATION_RMSE = 1
    MODEL_EVIDENCE = 2


if __name__ == "__main__":
    print(ValidationType.get_name(ValidationType.NEG_LOG_LIKELI))
