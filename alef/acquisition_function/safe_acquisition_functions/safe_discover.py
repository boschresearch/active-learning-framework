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
from alef.acquisition_function.safe_acquisition_functions.base_safe_acquisition_function import (
    StandardAlphaAcquisitionFunction,
)
from alef.acquisition_function.al_acquisition_functions.base_al_acquisition_function import BaseALAcquisitionFunction
from alef.acquisition_function.bo_acquisition_functions.base_bo_acquisition_function import BaseBOAcquisitionFunction
from alef.acquisition_function.al_acquisition_functions.pred_variance import PredVariance
from alef.models.base_model import BaseModel
from alef.acquisition_function.safe_acquisition_functions.utils import (
    compute_safe_probability,
    count_number_of_points_nearby,
    get_safety_models,
)


class SafeDiscover(StandardAlphaAcquisitionFunction):
    def __init__(
        self,
        safety_thresholds_lower: Union[float, Sequence[float]] = -np.inf,
        safety_thresholds_upper: Union[float, Sequence[float]] = np.inf,
        alpha: float = None,
        acquisition_function: Union[BaseALAcquisitionFunction, BaseBOAcquisitionFunction] = PredVariance(),
        **kwargs,
    ):
        super().__init__(safety_thresholds_lower, safety_thresholds_upper, alpha=alpha)
        self.__acq = acquisition_function

    def acquisition_score(
        self,
        x_grid: np.ndarray,
        model: BaseModel,
        *,
        safety_models: Optional[Sequence[BaseModel]] = None,
        x_data: np.ndarray = None,
        y_data: np.ndarray = None,
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
        print(f"{self.__class__.__name__} is under development.")
        S = self.compute_safe_set(x_grid, get_safety_models(model, safety_models))

        score = -np.inf * np.ones_like(S, dtype=float)

        acq_score = self.__acq.acquisition_score(x_grid[S], model, **kwargs)
        score_safe = np.zeros_like(acq_score, dtype=float)

        training_points_around = count_number_of_points_nearby(x_grid[S], model, safety_models, x_data, y_data)
        mask = training_points_around == min(training_points_around)

        if not np.any(mask):
            score[S] = 0

            if return_safe_set:
                return score, S
            else:
                return score

        score_safe[mask] = compute_safe_probability(
            x_grid[S][mask], self.safety_thresholds_lower, self.safety_thresholds_upper, model, safety_models
        )
        score[S] = score_safe

        if return_safe_set:
            return score, S
        else:
            return score


class SafeDiscoverQuantile(StandardAlphaAcquisitionFunction):
    def __init__(
        self,
        safety_thresholds_lower: Union[float, Sequence[float]] = -np.inf,
        safety_thresholds_upper: Union[float, Sequence[float]] = np.inf,
        alpha: float = None,
        acquisition_function: Union[BaseALAcquisitionFunction, BaseBOAcquisitionFunction] = PredVariance(),
        score_threshold_quantile: float = 0.5,
        **kwargs,
    ):
        super().__init__(safety_thresholds_lower, safety_thresholds_upper, alpha=alpha)
        self.__acq = acquisition_function

        assert score_threshold_quantile >= 0
        assert score_threshold_quantile <= 1

        self.__score_threshold_quantile = score_threshold_quantile

    def acquisition_score(
        self,
        x_grid: np.ndarray,
        model: BaseModel,
        *,
        safety_models: Optional[Sequence[BaseModel]] = None,
        x_data: np.ndarray = None,
        y_data: np.ndarray = None,
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
        print(f"{self.__class__.__name__} is under development.")
        S = self.compute_safe_set(x_grid, get_safety_models(model, safety_models))

        acq_score = self.__acq.acquisition_score(x_grid[S], model, **kwargs)
        q = np.quantile(acq_score, self.__score_threshold_quantile)
        mask = acq_score >= q

        score = -np.inf * np.ones_like(S, dtype=float)

        if not np.any(mask):
            score[S] = 0
            if return_safe_set:
                return score, S
            else:
                return score

        score_safe = np.zeros_like(acq_score, dtype=float)
        score_safe[mask] = compute_safe_probability(
            x_grid[S][mask], self.safety_thresholds_lower, self.safety_thresholds_upper, model, safety_models
        )

        score[S] = score_safe

        if return_safe_set:
            return score, S
        else:
            return score


if __name__ == "__main__":
    pass
