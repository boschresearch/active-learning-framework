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
    StandardBetaAcquisitionFunction,
)
from alef.acquisition_function.safe_acquisition_functions.safe_opt import SafeOpt
from alef.models.base_model import BaseModel
from alef.acquisition_function.safe_acquisition_functions.utils import (
    compute_safe_probability,
    count_number_of_points_nearby,
)


class SafeDiscoverOpt(StandardBetaAcquisitionFunction):
    def __init__(
        self,
        safety_thresholds_lower: Union[float, Sequence[float]] = -np.inf,
        safety_thresholds_upper: Union[float, Sequence[float]] = np.inf,
        beta: float = None,
        **kwargs,
    ):
        super().__init__(safety_thresholds_lower, safety_thresholds_upper, beta=beta)
        self.__acq = SafeOpt(
            safety_thresholds_lower=safety_thresholds_lower, safety_thresholds_upper=safety_thresholds_upper, beta=beta
        )

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
        acq_score, S = self.__acq.acquisition_score(
            x_grid, model, safety_models=safety_models, x_data=x_data, y_data=y_data, return_safe_set=True
        )
        MG = S > 1

        score = -1 * np.ones_like(S, dtype=float)
        score[MG] = 0

        training_points_around = count_number_of_points_nearby(x_grid[MG], model, safety_models, x_data, y_data)
        mask = training_points_around == min(training_points_around)

        if not np.any(mask):
            if return_safe_set:
                return score, S
            else:
                return score

        score_MG = np.zeros_like(score[MG], dtype=float)
        score_MG[mask] = compute_safe_probability(
            x_grid[MG][mask], self.safety_thresholds_lower, self.safety_thresholds_upper, model, safety_models
        )
        score[MG] = score_MG

        if return_safe_set:
            return score, S
        else:
            return score


class SafeDiscoverOptQuantile(StandardBetaAcquisitionFunction):
    def __init__(
        self,
        safety_thresholds_lower: Union[float, Sequence[float]] = -np.inf,
        safety_thresholds_upper: Union[float, Sequence[float]] = np.inf,
        beta: float = None,
        score_threshold_quantile: float = 0.5,
        **kwargs,
    ):
        super().__init__(safety_thresholds_lower, safety_thresholds_upper, beta=beta)
        self.__acq = SafeOpt(
            safety_thresholds_lower=safety_thresholds_lower, safety_thresholds_upper=safety_thresholds_upper, beta=beta
        )

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
        acq_score, S = self.__acq.acquisition_score(
            x_grid, model, safety_models=safety_models, x_data=x_data, y_data=y_data, return_safe_set=True
        )
        MG = S > 1

        q = np.quantile(acq_score[MG], self.__score_threshold_quantile)
        mask = acq_score[MG] >= q

        score = -1 * np.ones_like(S, dtype=float)
        score[MG] = 0

        training_points_around = count_number_of_points_nearby(x_grid[MG], model, safety_models, x_data, y_data)
        mask = training_points_around == min(training_points_around)

        if not np.any(mask):
            if return_safe_set:
                return score, S
            else:
                return score

        score_MG = np.zeros_like(score[MG], dtype=float)
        score_MG[mask] = compute_safe_probability(
            x_grid[MG][mask], self.safety_thresholds_lower, self.safety_thresholds_upper, model, safety_models
        )
        score[MG] = score_MG

        if return_safe_set:
            return score, S
        else:
            return score


if __name__ == "__main__":
    pass
