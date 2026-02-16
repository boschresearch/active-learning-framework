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
from scipy.stats import norm
from alef.acquisition_function.bo_acquisition_functions.base_bo_acquisition_function import BaseBOAcquisitionFunction
from alef.models.base_model import BaseModel


class EI(BaseBOAcquisitionFunction):
    """Expected improvement query strategy."""

    def __init__(self, xi: float = 0.0, **kwargs):
        """Init."""
        self.xi = xi

    def acquisition_score(
        self, x_grid: np.ndarray, model: BaseModel, x_data: np.ndarray, y_data: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """
        x_grid: [N, D] array, location for which acquisiton score should be calcluated
        model: BaseModel, surrogate model used to calculate acquisition score
        x_data: input locations of the datapoints already observed by the model - can be used to deduce current posterior of maximum (of latent f)
        y_data: ouput values of datapoints already observed by the model - can be used to determine currently observed maximum y value

        return:
            [N, ] array, acquisition score (later maximize this to get query point)
        """
        # deduce posterior of current maxmium of latent f on observed locations
        pred_mu_data, _ = model.predictive_dist(x_data)
        current_max = np.max(np.squeeze(pred_mu_data))
        # create predictive distribution on grid points
        pred_mu, pred_sigma = model.predictive_dist(x_grid)
        pred_mu = np.squeeze(pred_mu)
        pred_sigma = np.squeeze(pred_sigma)
        # calculate EI scores on grid locations
        d = pred_mu - current_max - self.xi
        score = d * norm.cdf(d / pred_sigma) + pred_sigma * norm.pdf(d / pred_sigma)
        return score


if __name__ == "__main__":
    pass
