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
from pydantic import BaseModel
from alef.acquisition_function.bo_acquisition_functions.base_bo_acquisition_function import BaseBOAcquisitionFunction
import numpy as np
from scipy.stats import norm
from alef.models.bayesian_ensemble_interface import BayesianEnsembleInterface


class IntegratedEI(BaseBOAcquisitionFunction):
    """Integrated Expected Improvement query strategy."""

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
        assert isinstance(model, BayesianEnsembleInterface)
        pred_dists_data = model.get_predictive_distributions(x_data)
        pred_dists_grid = model.get_predictive_distributions(x_grid)
        acquisition_functions_per_sample = []
        for i, pred_dist_grid in enumerate(pred_dists_grid):
            pred_mu_grid, pred_sigma_grid = pred_dist_grid
            pred_mu_data, _ = pred_dists_data[i]
            current_max = np.max(pred_mu_data)
            acquisition_function_per_sample = self.expected_improvement(current_max, pred_mu_grid, pred_sigma_grid)
            acquisition_functions_per_sample.append(acquisition_function_per_sample)
        score = np.mean(np.array(acquisition_functions_per_sample), axis=0)
        return score

    def expected_improvement(self, current_max, pred_mu_grid, pred_sigma_grid):
        d = pred_mu_grid - current_max - self.xi
        acquisition_function = d * norm.cdf(d / pred_sigma_grid) + pred_sigma_grid * norm.pdf(d / pred_sigma_grid)
        return acquisition_function
