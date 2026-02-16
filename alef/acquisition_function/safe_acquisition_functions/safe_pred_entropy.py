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

from typing import Sequence, Optional
import numpy as np
from alef.acquisition_function.safe_acquisition_functions.base_safe_acquisition_function import (
    StandardAlphaAcquisitionFunction,
)
from alef.models.base_model import BaseModel
from alef.acquisition_function.safe_acquisition_functions.utils import compute_gp_posterior
from alef.acquisition_function.safe_acquisition_functions.utils import (
    get_safety_models,
)


class SafePredEntropy(StandardAlphaAcquisitionFunction):
    def acquisition_score(
        self,
        x_grid: np.ndarray,
        model: BaseModel,
        *,
        safety_models: Optional[Sequence[BaseModel]] = None,
        return_safe_set: bool = False,
        **kwargs,
    ):
        r"""
        x_grid: [N, D] array, location for which acquisiton score should be calcluated
        model: BaseModel, surrogate model used to calculate acquisition score
        safety_models: None or list of BaseModel classes, surrogate models for safety functions
                        if None then one may consider model as the safety model as well

        return:
            if return_safe_set is True:
                [N, ] array, acquisition score (later maximize this to get query point)
                [N, ] array, bool, True for safe points and False for unsafe points
            if return_safe_set is False:
                [N, ] array, acquisition score (later maximize this to get query point)
        """
        S = self.compute_safe_set(x_grid, get_safety_models(model, safety_models))
        score = -np.inf * np.ones_like(S, dtype=float)

        _, pred_sigma = model.predictive_dist(x_grid[S])
        pred_sigma = np.squeeze(pred_sigma)

        score[S] = 0.5 * np.log((2 * np.pi * np.e) * pred_sigma**2)
        if return_safe_set:
            return score, S
        else:
            return score


class SafePredEntropyAll(StandardAlphaAcquisitionFunction):
    def acquisition_score(
        self,
        x_grid: np.ndarray,
        model: BaseModel,
        safety_models: Optional[Sequence[BaseModel]] = None,
        return_safe_set: bool = False,
        **kwargs,
    ):
        r"""
        x_grid: [N, D] array, location for which acquisiton score should be calcluated
        model: BaseModel, surrogate model of main function
        safety_models: None or list of BaseModel classes, surrogate models for safety functions
                        if None then one may consider model as the safety model as well

        return:
            if return_safe_set is True:
                [N, ] array, acquisition score (later maximize this to get query point)
                [N, ] array, bool, True for safe points and False for unsafe points
            if return_safe_set is False:
                [N, ] array, acquisition score (later maximize this to get query point)
        """
        S = self.compute_safe_set(x_grid, get_safety_models(model, safety_models))
        score = -np.inf * np.ones_like(S, dtype=float)

        _, pred_sigma = compute_gp_posterior(x_grid[S], model, safety_models)
        entropy = 0.5 * np.log((2 * np.pi * np.e) * pred_sigma**2)
        score[S] = np.sum(entropy, axis=1)

        if return_safe_set:
            return score, S
        else:
            return score


if __name__ == "__main__":
    pass
