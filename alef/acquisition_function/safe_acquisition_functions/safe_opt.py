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

from typing import Sequence, List, Optional
import numpy as np
from alef.acquisition_function.safe_acquisition_functions.base_safe_acquisition_function import (
    StandardBetaAcquisitionFunction,
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


class SafeOpt(StandardBetaAcquisitionFunction):
    def acquisition_score(
        self,
        x_grid: np.ndarray,
        model: BaseModel,
        *,
        safety_models: Optional[Sequence[BaseModel]] = None,
        x_data: Optional[np.ndarray] = None,
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
        Q = compute_confidence_intervals(x_grid, self.beta, model, safety_models)

        M, G = self.compute_sets(x_grid, S, Q, get_safety_models(model, safety_models), False)

        MG = np.logical_or(M, G)
        l = Q[:, ::2]
        u = Q[:, 1::2]

        score = -np.inf * np.ones_like(S)
        score[MG] = np.max(u[MG] - l[MG], axis=1)

        if return_safe_set:
            return score, np.where(MG, 2, S.astype(int))
        else:
            return score

    def compute_sets(
        self, x_grid: np.ndarray, S: np.ndarray, Q: np.ndarray, safety_models: List, full_sets: bool = False
    ):
        """
        Compute the safe set of points, based on current confidence bounds.
        Parameters
        ----------
        x_grid: [N, D] array, candidate points
        S: [N,] array, safety mask
        Q: [N, Q] array, confidence interval from all models
            Q is rescaled so each model has uncertainty in the same scale
        safety_models: list of safety models
        full_sets: boolean
            Whether to compute the full set of expanders or whether to omit
            computations that are not relevant for running SafeOpt
            (This option is only useful for plotting purposes)
        """
        sqrt_beta = np.sqrt(self.beta)
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
            # Check if expander for all GPs
            l = l[..., -len(safety_models) :]
            u = u[..., -len(safety_models) :]
            for i, model in enumerate(safety_models):
                # Skip evlauation if 'no' safety constraint
                if self.safety_thresholds_lower[i] == -np.inf and self.safety_thresholds_upper[i] == np.inf:
                    continue

                if True:  # bad format, but don't see how to improve yet
                    # Add safe point with its max possible value to the model
                    x_data, f = model.model.data

                    x_pseudo = np.vstack((x_data, np.atleast_2d(x_grid[s, :][index, :])))
                    if np.shape(f)[1] == 2:
                        p = np.atleast_2d(x_grid[s, :][index, -1])

                    f_pseudo_for_l = np.atleast_2d(
                        u[s, i][index]
                    )  # optimistic upper bound to see if more points meet lower bound
                    f_pseudo_for_u = np.atleast_2d(
                        l[s, i][index]
                    )  # optimistic lower bound to see if more points meet upper bound
                    if np.shape(f)[1] == 2:
                        f_pseudo_for_l = np.hstack((f_pseudo_for_l, p))
                        f_pseudo_for_u = np.hstack((f_pseudo_for_u, p))

                    f_pseudo = np.vstack((f, f_pseudo_for_l))
                    # in the following, pred_sigma is not rescaled, which is what we need to compare to safety thresholds
                    model.model.data = (x_pseudo, f_pseudo)
                    # Prediction of previously unsafe points based on that
                    pred_mu, pred_sigma = model.predictive_dist(x_grid[~S])
                    l2 = pred_mu - sqrt_beta * pred_sigma

                    f_pseudo = np.vstack((f, f_pseudo_for_u))
                    model.model.data = (x_pseudo, f_pseudo)
                    # Prediction of previously unsafe points based on that
                    pred_mu, pred_sigma = model.predictive_dist(x_grid[~S])
                    u2 = pred_mu + sqrt_beta * pred_sigma

                    # Remove the fake data point from the GP again
                    model.model.data = (x_data, f)

                # If any unsafe lower bound is suddenly above fmin
                # or unsafe upper bound below fmax
                # then the point is an expander
                G_safe[index] = np.logical_and(
                    np.any(l2 >= self.safety_thresholds_lower[i]), np.any(u2 <= self.safety_thresholds_upper[i])
                )

                # Break if one safety GP is not an expander
                if not G_safe[index]:
                    break

            # Since we sorted by uncertainty and only the most
            # uncertain element gets picked by SafeOpt anyways, we can
            # stop after we found the first one
            if G_safe[index] and not full_sets:
                break

        # Update safe set (if full_sets is False this is at most one point
        G[s] = G_safe
        return M, G


if __name__ == "__main__":
    pass
