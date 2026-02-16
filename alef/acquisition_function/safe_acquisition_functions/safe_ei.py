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

from typing import Union, Sequence, Optional
import numpy as np
from scipy.stats import norm
from alef.acquisition_function.safe_acquisition_functions.base_safe_acquisition_function import (
    StandardAlphaAcquisitionFunction,
)
from alef.models.base_model import BaseModel
from alef.acquisition_function.safe_acquisition_functions.utils import (
    get_safety_models,
)


# expected improvement
class SafeEI(StandardAlphaAcquisitionFunction):
    def __init__(
        self,
        safety_thresholds_lower: Union[float, Sequence[float]] = -np.inf,
        safety_thresholds_upper: Union[float, Sequence[float]] = np.inf,
        alpha: float = None,
        xi: float = None,
        **kwargs,
    ):
        super().__init__(safety_thresholds_lower, safety_thresholds_upper, alpha=alpha)
        self.xi = xi

    def acquisition_score(
        self,
        x_grid: np.ndarray,
        model: BaseModel,
        *,
        safety_models: Optional[Sequence[BaseModel]] = None,
        x_data: np.ndarray = None,
        y_data: Optional[np.ndarray] = None,
        return_safe_set: bool = False,
        **kwargs,
    ):
        r"""
        x_grid: [N, D] array, location for which acquisiton score should be calcluated
        model: BaseModel, surrogate model used to calculate acquisition score
        safety_models: None or list of BaseModel classes, surrogate models for safety functions
                        if None then one may consider model as the safety model as well
        x_data: input locations of the datapoints already observed by the model - can be used to deduce current posterior of maximum (of latent f)
        y_data: ouput values of datapoints already observed by the model - can be used to determine currently observed maximum y value

        return:
            if return_safe_set is True:
                [N, ] array, acquisition score (later maximize this to get query point)
                [N, ] array, bool, True for safe points and False for unsafe points
            if return_safe_set is False:
                [N, ] array, acquisition score (later maximize this to get query point)
        """
        S = self.compute_safe_set(x_grid, safety_models=get_safety_models(model, safety_models))

        # deduce posterior of current maxmium of latent f on observed locations
        pred_mu_data, _ = model.predictive_dist(x_data)
        current_max = np.max(np.squeeze(pred_mu_data))

        # create predictive distribution on grid points
        pred_mu, pred_sigma = model.predictive_dist(x_grid)
        pred_mu = np.squeeze(pred_mu)
        pred_sigma = np.squeeze(pred_sigma)

        # calculate EI scores on grid locations
        d = pred_mu - current_max - self.xi

        score = -np.inf * np.ones_like(S)
        score[S] = d * norm.cdf(d / pred_sigma) + pred_sigma * norm.pdf(d / pred_sigma)

        if return_safe_set:
            return score, S
        else:
            return score


if __name__ == "__main__":
    pass
