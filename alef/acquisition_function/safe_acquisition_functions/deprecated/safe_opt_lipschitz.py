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
from scipy.spatial.distance import cdist
from alef.acquisition_function.safe_acquisition_functions.base_safe_acquisition_function import (
    StandardSafeAcquisitionFunction,
)
from alef.models.base_model import BaseModel
from alef.acquisition_function.safe_acquisition_functions.utils import (
    sort_generator,
    get_safety_models,
    compute_confidence_intervals,
)

"""
Some pieces of the following class ConstrainedBayesianOptimizer is adapted from SAFEOPT
(https://github.com/befelix/SafeOpt/blob/master/safeopt/gp_opt.py
Licensed under the MIT License).
"""


class SafeOptLipschitz(StandardSafeAcquisitionFunction):
    def __init__(
        self,
        safety_thresholds_lower: Union[float, Sequence[float]] = -np.inf,
        safety_thresholds_upper: Union[float, Sequence[float]] = np.inf,
        beta: float = None,
        lipschitz: Union[float, Sequence[float]] = None,
        **kwargs,
    ):
        print("please do not use this method when the GP model is updated in every iteration")
        super().__init__(safety_thresholds_lower, safety_thresholds_upper, beta=beta)
        self.__set_lipschitz(lipschitz)
        self.S = None

    def acquisition_score(
        self,
        x_grid: np.ndarray,
        model: BaseModel,
        safety_models: Optional[Sequence[BaseModel]] = None,
        x_data: Optional[np.ndarray] = None,
        y_data: Optional[np.ndarray] = None,
        S_current: np.ndarray = None,
        return_safe_set: bool = False,
        **kwargs,
    ):
        r"""
        x_grid: [N, D] array, location for which acquisiton score should be calcluated
        model: BaseModel, surrogate model used to calculate acquisition score
        safety_models: None or list of BaseModel classes, surrogate models for safety functions
        x_data: input locations of the datapoints already observed by the model - can be used to deduce current posterior of maximum (of latent f)
        y_data: ouput values of datapoints already observed by the model - can be used to determine currently observed maximum y value

        return:
            if return_safe_set is True:
                [N, ] array, acquisition score (later maximize this to get query point)
                [N, ] array, bool, True for safe points and False for unsafe points
            if return_safe_set is False:
                [N, ] array, acquisition score (later maximize this to get query point)
        """
        print("please do not use this method when the GP model is updated in every iteration")
        S = self.compute_safe_set(x_grid, safety_models=get_safety_models(model, safety_models), S_current=S_current)
        Q = compute_confidence_intervals(x_grid, self.beta, model, safety_models)

        M, G = self.compute_sets(x_grid, S, Q, False)

        MG = np.logical_or(M, G)

        l = Q[:, ::2]
        u = Q[:, 1::2]

        score = -np.inf * np.ones_like(S)
        score[MG] = np.max(u[MG] - l[MG], axis=1)

        if return_safe_set:
            return score, np.where(MG, 2, S.astype(int))
        else:
            return score

    def compute_sets(self, x_grid: np.ndarray, S: np.ndarray, Q: np.ndarray, full_sets: bool = False):
        """
        Compute the safe set of points, based on current confidence bounds.
        Parameters
        ----------
        x_grid: [N, D] array, candidate points
        S: [N,] array, safety mask
        Q: [N, Q] array, confidence interval from all models
        full_sets: boolean
            Whether to compute the full set of expanders or whether to omit
            computations that are not relevant for running SafeOpt
            (This option is only useful for plotting purposes)

        return:
        M - [N,] bool array, maximizers
        G - [N,] bool array, safe set expanders
        """
        np.sqrt(self.beta)
        if not full_sets:
            print("only compute safe set expander which has max uncertainty (this is faster)")

        # Reference to confidence intervals
        l, u = Q[:, :2].T

        # Set of possible maximisers
        # Maximizers: safe upper bound above best, safe lower bound
        M = np.zeros_like(S, dtype=bool)
        M[S] = u[S] >= np.max(l[S])
        max_var = np.max(u[M] - l[M])

        # Optimistic set of possible expanders
        l = Q[:, ::2]
        u = Q[:, 1::2]

        G = np.zeros_like(S, dtype=bool)

        # For the run of the algorithm we do not need to calculate the
        # full set of potential expanders:
        # We can skip the ones already in M and ones that have lower
        # variance than the maximum variance in M, max_var or the threshold.
        # Amongst the remaining ones we only need to find the
        # potential expander with maximum variance
        if full_sets:
            s = S
        else:
            # skip points in M, they will already be evaluated (the query is in set union(G, M), so M is there anyway)
            s = np.logical_and(S, ~M)

            # Remove points with a variance that is too small
            s[s] = np.max((u[s, :] - l[s, :]), axis=1) > max_var

            if not np.any(s):
                # no need to evaluate any points as expanders in G, exit
                return M, G

        # set of safe expanders
        G_safe = np.zeros(np.count_nonzero(s), dtype=np.bool)

        if not full_sets:
            # Sort, element with largest variance first
            sort_index = sort_generator(np.max(u[s, :] - l[s, :], axis=1))
        else:
            # Sort index is just an enumeration of all safe states
            sort_index = range(len(G_safe))

        for index in sort_index:
            # Distance between current index point and all other unsafe
            # points
            d = cdist(x_grid[s, :][[index], :], x_grid[~S, :])  # [1, #~S]
            lips = np.reshape(self.lipschitz, [-1, 1])
            QL = Q[s, ::2][index, -self.number_of_constraints :, None]  # [#constraints, 1]
            QH = Q[s, 1::2][index, -self.number_of_constraints :, None]  # [#constraints, 1]

            mask_low = np.all(
                (QL - lips * d) >= np.reshape(self.safety_thresholds_lower, [-1, 1]), axis=0
            )  # [#~S] all safe lower bounds checked
            mask_upp = np.all(
                (QH + lips * d) <= np.reshape(self.safety_thresholds_upper, [-1, 1]), axis=0
            )  # [#~S] all safe upper bounds checked

            mask = np.logical_and(mask_low, mask_upp)
            G_safe[index] = np.any(mask)
            """
            # Check if expander for all GPs
            for i in range(self.number_of_constraints):
                # Skip evaluation if 'no' safety constraint
                if self.safety_thresholds_lower[i] == -np.inf and self.safety_thresholds_upper[i] == np.inf:
                    continue
                # the safe point(s) that has surrounding points which could expand the safe set:
                # u - L * d >= safety_thresholds_lower
                idx = i if Q.shape[1] == 2*len(self.safety_thresholds_lower) else i + 1
                G_safe[index] = np.any(
                    np.logical_and(
                        (u[s, idx][index] - self.lipschitz[i] * d) >= self.safety_thresholds_lower[i],
                        (u[s, idx][index] + self.lipschitz[i] * d) <= self.safety_thresholds_upper[i]
                    )
                )

                # Stop evaluating if not expander according to one
                # safety constraint
                if not G_safe[index]:
                    break
            """
            # Since we sorted by uncertainty and only the most
            # uncertain element gets picked by SafeOpt anyways, we can
            # stop after we found the first one
            if G_safe[index] and not full_sets:
                break

        # Update safe set (if full_sets is False this is at most one point
        G[s] = G_safe
        return M, G

    def initialize_safe_set(self, x_grid: np.ndarray, safety_models: Sequence[BaseModel], **kwargs):
        r"""
        x_grid: [N, D] array, location for which safety should be classified
        safety_models: None or list of BaseModel classes, surrogate models for safety functions
        **kwargs: additional arguments or unused arguments

        return:
            [N, ] array, bool, True for safe points and False for unsafe points
        """
        safety_lower = self.safety_thresholds_lower
        safety_upper = self.safety_thresholds_upper
        Q = compute_confidence_intervals(x_grid, self.beta, None, safety_models)

        S = np.all(Q[:, ::2] >= safety_lower, axis=1) * np.all(Q[:, 1::2] <= safety_upper, axis=1)
        return S

    def compute_safe_set(
        self, x_grid: np.ndarray, safety_models: Sequence[BaseModel], S_current: np.ndarray = None, **kwargs
    ):
        r"""
        x_grid: [N, D] array, location for which safety should be classified
        safety_models: None or list of BaseModel classes, surrogate models for safety functions
        **kwargs: additional arguments or unused arguments

        return:
            [N, ] array, bool, True for safe points and False for unsafe points
        """
        print(
            "Do not use this method when the model is updated in every iteration."
            + "SafeOpt fix GP hyperparameters and expand with new observations and Lipschitz constants."
            + "If we keep updating the GP hyperpars, S would be changing,"
            + "meaning that previous safe points may become unsafe now with new model."
            + "Expanding from previous set just wont be correct then."
        )
        if S_current is None:
            S = self.initialize_safe_set(x_grid, safety_models)
        else:
            S = S_current.copy()
        if np.all(S):
            return S

        Q = compute_confidence_intervals(x_grid, self.beta, None, safety_models)

        d = cdist(x_grid[S, :], x_grid[~S, :])[:, None, :]
        lips = np.reshape(self.lipschitz, [1, -1, 1])

        mask_low = np.all(
            (Q[S, ::2, None] - lips * d) >= np.reshape(self.safety_thresholds_lower, [1, -1, 1]), axis=1
        )  # [#S, #~S] all safe lower bounds checked
        mask_upp = np.all(
            (Q[S, 1::2, None] + lips * d) <= np.reshape(self.safety_thresholds_upper, [1, -1, 1]), axis=1
        )  # [#S, #~S] all safe upper bounds checked
        mask = np.logical_and(mask_low, mask_upp)

        S[~S] = np.logical_or(S[~S], np.any(mask, axis=0))
        return S

    def __set_lipschitz(self, lipschitz: Union[float, Sequence[float]]):
        self.lipschitz = None
        if not lipschitz is None:
            if isinstance(lipschitz, list):
                assert len(lipschitz) == len(self.safety_thresholds_lower)
                self.lipschitz = np.atleast_1d(np.asarray(lipschitz).squeeze())
            else:
                self.lipschitz = np.atleast_1d(np.asarray([lipschitz] * self.number_of_constraints).squeeze())


if __name__ == "__main__":
    pass
