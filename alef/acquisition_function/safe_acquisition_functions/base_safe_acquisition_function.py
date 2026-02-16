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

from typing import Tuple, Union, Sequence, Optional
from abc import abstractmethod
import numpy as np
from scipy.stats import norm
from alef.models.base_model import BaseModel


class BaseSafeAcquisitionFunction:
    @abstractmethod
    def acquisition_score(
        self,
        x_grid: np.ndarray,
        model: BaseModel,
        *,
        safety_models: Optional[Sequence[BaseModel]],
        x_data: Optional[np.ndarray],
        y_data: Optional[np.ndarray],
        return_safe_set: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        x_grid: [N, D] array, location for which acquisiton score should be calcluated
        model: BaseModel, surrogate model used to calculate acquisition score
        safety_models: None or list of BaseModel classes, surrogate models for safety functions
                        if None then one may consider model as the safety model as well
        x_data: input locations of the datapoints already observed by the model - can be used to deduce current posterior of maximum (of latent f)
        y_data: ouput values of datapoints already observed by the model - can be used to determine currently observed maximum y value
        return_safe_set: return safety mask or not

        return:
            if return_safe_set is True:
                [N, ] array, acquisition score (later maximize this to get query point)
                [N, ] array, bool, True for safe points and False for unsafe points
            if return_safe_set is False:
                [N, ] array, acquisition score (later maximize this to get query point)

        """
        raise NotImplementedError

    @abstractmethod
    def compute_safe_set(self, x_grid: np.ndarray, safety_models: Sequence[BaseModel], **kwargs) -> np.ndarray:
        r"""
        x_grid: [N, D] array, location for which safety should be classified
        safety_models: list of BaseModel classes, surrogate models for safety functions
        **kwargs: additional arguments or unused arguments

        return:
            [N, ] array, bool, True for safe points and False for unsafe points
        """
        raise NotImplementedError

    def compute_safe_data_set(self, Z: np.ndarray) -> np.ndarray:
        r"""
        Z: [N, P] array, observed safety values

        return:
            [N, ] array, bool, True for safe points and False for unsafe points
        """
        raise NotImplementedError


class StandardSafeAcquisitionFunction(BaseSafeAcquisitionFunction):
    def __init__(
        self,
        safety_thresholds_lower: Union[float, Sequence[float]] = -np.inf,
        safety_thresholds_upper: Union[float, Sequence[float]] = np.inf,
        alpha: float = None,
        beta: float = None,
        **kwargs,
    ):
        assert (not alpha is None) or (not beta is None)

        self.__lower = self.__set1Dlist(safety_thresholds_lower)
        self.__upper = self.__set1Dlist(safety_thresholds_upper)
        assert len(self.__lower) == len(self.__upper)
        self.__constraints_num = len(self.__lower)

        if not alpha is None:
            self.alpha = alpha

        if not beta is None:
            self.beta = beta

    def compute_safe_data_set(self, Z: np.ndarray):
        r"""
        Z: [N, P] array, observed safety values

        return:
            [N, ] array, bool, True for safe points and False for unsafe points
        """
        return np.all((Z <= self.safety_thresholds_upper) * (Z >= self.safety_thresholds_lower), axis=1)

    @property
    def number_of_constraints(self):
        return self.__constraints_num

    @property
    def safety_thresholds_lower(self):
        return self.__lower

    @safety_thresholds_lower.setter
    def safety_thresholds_lower(self, thresholds: Union[float, Sequence[float]]):
        self.__lower = self.__set1Dlist(thresholds)
        assert len(self.__lower) == self.number_of_constraints

    @property
    def safety_thresholds_upper(self):
        return self.__upper

    @safety_thresholds_upper.setter
    def safety_thresholds_upper(self, thresholds: Union[float, Sequence[float]]):
        self.__upper = self.__set1Dlist(thresholds)
        assert len(self.__upper) == self.number_of_constraints

    def __set1Dlist(self, variables: Union[float, Sequence[float]]):
        return np.reshape(variables, -1).tolist()


class StandardAlphaAcquisitionFunction(StandardSafeAcquisitionFunction):
    def __init__(
        self,
        safety_thresholds_lower: Union[float, Sequence[float]] = -np.inf,
        safety_thresholds_upper: Union[float, Sequence[float]] = np.inf,
        alpha: float = None,
        **kwargs,
    ):
        super().__init__(safety_thresholds_lower, safety_thresholds_upper, alpha=alpha, beta=None)

    def compute_safe_set(self, x_grid: np.ndarray, safety_models: Sequence[BaseModel], **kwargs) -> np.ndarray:
        r"""
        x_grid: [N, D] array, location for which safety should be classified
        safety_models: list of BaseModel classes, surrogate models for safety functions
        **kwargs: additional arguments or unused arguments

        return:
            [N, ] array, bool, True for safe points and False for unsafe points
        """
        safety_lower = self.safety_thresholds_lower
        safety_upper = self.safety_thresholds_upper

        len(safety_models)
        N = np.shape(x_grid)[0]
        S = np.ones(N, dtype=np.bool)

        for i, model in enumerate(safety_models):
            mu, sigma = model.predictive_dist(x_grid)
            S = S * self.__1d_safety_check(safety_lower[i], safety_upper[i], self.alpha, mu, sigma)

        return S

    def __1d_safety_check(self, threshold_lower: float, threshold_upper: float, alpha: float, pred_mu, pred_sigma):
        assert threshold_upper >= threshold_lower

        prob_below_lower = norm.cdf(threshold_lower, pred_mu, pred_sigma)
        prob_below_upper = norm.cdf(threshold_upper, pred_mu, pred_sigma)

        return np.reshape((prob_below_upper - prob_below_lower) >= 1 - alpha, -1)


class StandardBetaAcquisitionFunction(StandardSafeAcquisitionFunction):
    def __init__(
        self,
        safety_thresholds_lower: Union[float, Sequence[float]] = -np.inf,
        safety_thresholds_upper: Union[float, Sequence[float]] = np.inf,
        beta: float = None,
        **kwargs,
    ):
        super().__init__(safety_thresholds_lower, safety_thresholds_upper, beta=beta)

    def compute_safe_set(self, x_grid: np.ndarray, safety_models: Sequence[BaseModel], **kwargs) -> np.ndarray:
        r"""
        x_grid: [N, D] array, location for which safety should be classified
        safety_models: list of BaseModel classes, surrogate models for safety functions
        **kwargs: additional arguments or unused arguments

        return:
            [N, ] array, bool, True for safe points and False for unsafe points
        """
        self.safety_thresholds_lower
        self.safety_thresholds_upper

        sqrt_beta = np.sqrt(self.beta)

        len(safety_models)
        N = np.shape(x_grid)[0]
        S = np.ones(N, dtype=np.bool)

        for i, model in enumerate(safety_models):
            mu, sigma = model.predictive_dist(x_grid)

            ub = mu + sqrt_beta * sigma
            lb = mu - sqrt_beta * sigma

            S = (
                S
                * np.reshape(ub <= self.safety_thresholds_upper[i], -1)
                * np.reshape(lb >= self.safety_thresholds_lower[i], -1)
            )

        return S
