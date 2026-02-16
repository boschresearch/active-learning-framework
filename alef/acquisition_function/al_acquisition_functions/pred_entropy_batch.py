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

import numpy as np
from alef.acquisition_function.al_acquisition_functions.base_batch_al_acquisition_function import (
    BaseBatchALAcquisitionFunction,
)
from alef.models.batch_model_interface import BatchModelInterace


class PredEntropyBatch(BaseBatchALAcquisitionFunction):
    """Predictive entropy query strategy (batch)."""

    def __init__(self, **kwargs):
        """Init."""

    def acquisition_score(self, batch: np.ndarray, model: BatchModelInterace, **kwargs) -> np.float32:
        r"""In this interface this function only provides point calculation for one singe input (one batch) -> output is one number
        batch: [N, D] array, single batch for which acquisiton score should be calcluated
        model: BaseModel, surrogate model used to calculate acquisition score

        return:
            np.float - single acquisition score
        """
        return model.entropy_predictive_dist_full_cov(batch)


if __name__ == "__main__":
    pass
