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

from typing import Optional
import numpy as np
from alef.acquisition_function.bo_acquisition_functions.base_bo_acquisition_function import BaseBOAcquisitionFunction
from alef.models.base_model import BaseModel


class GPUCB(BaseBOAcquisitionFunction):
    """GP-UCB query strategy."""

    def __init__(self, beta: float = 3.0, **kwargs):
        """Init."""
        self.beta = beta

    def acquisition_score(
        self,
        x_grid: np.ndarray,
        model: BaseModel,
        x_data: Optional[np.ndarray] = None,
        y_data: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        x_grid: [N, D] array, location for which acquisiton score should be calcluated
        model: BaseModel, surrogate model used to calculate acquisition score
        x_data: input locations of the datapoints already observed by the model - can be used to deduce current posterior of maximum (of latent f)
        y_data: ouput values of datapoints already observed by the model - can be used to determine currently observed maximum y value

        return:
            [N, ] array, acquisition score (later maximize this to get query point)
        """
        pred_mu, pred_sigma = model.predictive_dist(x_grid)
        pred_mu = np.squeeze(pred_mu)
        pred_sigma = np.squeeze(pred_sigma)
        score = pred_mu + np.sqrt(self.beta) * pred_sigma
        return score


if __name__ == "__main__":
    pass
