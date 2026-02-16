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

import numpy as np
import scipy.spatial
from matplotlib import pyplot as plt
import copy
from typing import Union, List, Tuple

"""
This is from the attached code of paper
Jonas Rothfuss, Christopher Koenig, Alisa Rupenyan, Andreas Krause, CoRL 2022,
Meta-Learning Priors for Safe Bayesian Optimization

The link is published by the authors
https://tinyurl.com/safe-meta-bo

see folder ./meta_bo/bracketing.py

Copyright (c) 2022 Jonas Rothfuss, licensed under the MIT License

"""


class Frontier:
    def __init__(self, ndim: int, boundary: np.ndarray, lower: bool = False):
        self.ndim = ndim
        self.frontier_points = np.empty((0, ndim))
        assert boundary.shape == (self.ndim,)
        self.boundary = boundary
        self.lower = lower
        self.add(boundary)

    def add(self, point: np.ndarray) -> None:
        if point.ndim == 1:
            point = point.reshape((1, -1))
        assert point.shape[-1] == self.ndim
        self.frontier_points = np.concatenate([self.frontier_points, point])

    def frontier(self, dim: int, x: float) -> Union[np.ndarray, float]:
        if self.lower:
            valid_values = self.frontier_points[x <= self.frontier_points[:, dim]]
        else:
            valid_values = self.frontier_points[x >= self.frontier_points[:, dim]]
        assert self.ndim == 2
        if len(valid_values) == 0:
            return self.boundary[int(not dim)]
        else:
            if self.lower:
                return np.max(valid_values[:, int(not dim)])
            else:
                return np.min(valid_values[:, int(not dim)])


class MonotoneFrontierSolver:
    """
    Monotone frontier solver that tries to solve an optimization problem of the form

    min objective(x) s.t. constraint(x) <= 0

    It assumes that constraint(x) is a monotonically increasing function in all dimensions of x
    and that objective(x) is monotonically decreasing. It requires a lower and upper boundary
    for x and from there advances an a safe and an unsafe frontier that converge towards
    the curve objective(x) = 0 from above and below.
    """

    def __init__(self, ndim: int, lower_boundary: np.ndarray, upper_boundary: np.ndarray):
        if ndim != 2:
            raise NotImplementedError("MonotoneFrontierSolver currently only supports ndim = 2")
        self.ndim = ndim
        assert lower_boundary.shape == (ndim,) and upper_boundary.shape == (ndim,)
        assert np.all(lower_boundary < upper_boundary)
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary

        self.unsafe_frontier = Frontier(ndim=ndim, boundary=upper_boundary, lower=False)
        self.safe_frontier = Frontier(ndim=ndim, boundary=lower_boundary, lower=True)

        self.candidate_points = [(upper_boundary + lower_boundary) / 2]
        self.evaluations = []

    def score_candidate_points(self, candidate_points: List) -> List[float]:
        scores = []
        for x in candidate_points:
            expanded_area_unsafe = (self.unsafe_frontier.frontier(dim=0, x=x[0]) - x[1]) * (
                self.unsafe_frontier.frontier(dim=1, x=x[1]) - x[0]
            )
            expanded_area_safe = (x[1] - self.safe_frontier.frontier(dim=0, x=x[0])) * (
                x[0] - self.safe_frontier.frontier(dim=1, x=x[1])
            )
            assert expanded_area_safe >= 0 and expanded_area_unsafe >= 0
            scores.append(min(expanded_area_safe, expanded_area_unsafe))
        assert len(scores) == len(candidate_points)
        return scores

    def next(self) -> np.ndarray:
        """Select candidate point which guarantees the biggest expansion of either frontier."""
        candidate_points = [
            np.array([x0, (self.unsafe_frontier.frontier(0, x0) + self.safe_frontier.frontier(0, x0)) / 2])
            for x0 in np.linspace(self.lower_boundary[0], self.upper_boundary[0], num=max(100, self.num_evals * 10))
        ]
        candidate_scores = self.score_candidate_points(candidate_points)
        best_idx = np.argmax(candidate_scores)
        return candidate_points[best_idx]

    @property
    def safe_evaluations(self) -> List[Tuple[np.ndarray, float, float]]:
        return [(point, constr, objective) for point, constr, objective in self.evaluations if constr <= 0]

    @property
    def unsafe_evaluations(self) -> List[Tuple[np.ndarray, float, float]]:
        return [(point, constr, objective) for point, constr, objective in self.evaluations if constr > 0]

    @property
    def num_evals(self) -> int:
        return len(self.evaluations)

    @property
    def best_safe_evaluation(self) -> Tuple[np.ndarray, float, float]:
        safe_evals = self.safe_evaluations
        if len(safe_evals) > 0:
            best_idx = np.argmin([obj for _, _, obj in safe_evals])
            return safe_evals[best_idx]
        else:
            return self.lower_boundary, -np.inf, -np.inf

    def add_eval(self, point: np.ndarray, constr: float, objective: float) -> None:
        self.evaluations.append((point, constr, objective))
        if constr <= 0.0:
            self.safe_frontier.add(point)
            for j in range(self.ndim):
                new_point = copy.deepcopy(point)
                new_point[j] = (self.unsafe_frontier.frontier(int(not j), point[int(not j)]) + point[j]) / 2
                self.candidate_points.append(new_point)
        else:
            self.unsafe_frontier.add(point)
            for j in range(self.ndim):
                new_point = copy.deepcopy(point)
                new_point[j] = (self.safe_frontier.frontier(int(not j), point[int(not j)]) + point[j]) / 2
                self.candidate_points.append(new_point)


class MonotoneFrontierSolverV2:
    """
    Monotone frontier solver that tries to solve an optimization problem of the form

    min objective(x) s.t. constraint(x) >= 0

    It assumes that constraint(x) and objective(x) are monotonically increasing functions in all dimensions of x.
    It requires a lower and upper boundary for x and from there advances an a safe and an unsafe frontier that
    converge towards the curve objective(x) = 0 from above and below.
    """

    def __init__(self, ndim: int, lower_boundary: np.ndarray, upper_boundary: np.ndarray):
        if ndim != 2:
            raise NotImplementedError("MonotoneFrontierSolver currently only supports ndim = 2")
        self.ndim = ndim
        assert lower_boundary.shape == (ndim,) and upper_boundary.shape == (ndim,)
        assert np.all(lower_boundary < upper_boundary)
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary

        self.upper_left_corner = np.array([self.lower_boundary[0], self.upper_boundary[1]])
        self.lower_right_corner = np.array([self.upper_boundary[0], self.lower_boundary[1]])

        self.unsafe_frontier = Frontier(ndim=ndim, boundary=upper_boundary, lower=True)
        self.safe_frontier = Frontier(ndim=ndim, boundary=lower_boundary, lower=False)

        self.upper_frontier_points = self.upper_boundary[None, :]
        self.lower_frontier_points = self.lower_boundary[None, :]

        # add lower and upper boundary evals
        self.evaluations = [(self.lower_boundary, -np.inf, -np.inf), (self.upper_boundary, np.inf, np.inf)]
        self.t = 0

    def frontier_middle_points(self, n: int = 100):
        z1_arr = np.linspace(self.lower_boundary[0], self.upper_boundary[0], n)
        z2_middle = np.array([(self.upper_frontier(z) + self.lower_frontier(z)) / 2 for z in z1_arr])

        z2_arr = np.linspace(self.lower_boundary[1], self.upper_boundary[1], n)
        z1_middle = np.array([(self.upper_frontier(z, 1) + self.lower_frontier(z, 1)) / 2 for z in z2_arr])
        return np.concatenate([np.stack([z1_arr, z2_middle], axis=-1), np.stack([z1_middle, z2_arr], axis=-1)], axis=0)

    def next(self) -> np.ndarray:
        """Select candidate point which guarantees the biggest expansion of either frontier."""
        self.t += 1

        lower_corners = self.lower_corners_w_domain_corners
        upper_corners = self.upper_corners
        max_min_indices, dist = self.max_min_distance(lower_corners, upper_corners)
        assert len(max_min_indices) > 0

        rectangles = []
        max_min_dists = []
        for rect_u in self.upper_corners:
            for rect_l in self.lower_corners_w_domain_corners:
                if (
                    self.rectangle_is_between_frontiers(rect_u, rect_l)
                    and rect_l[0] <= rect_u[0]
                    and rect_l[1] <= rect_u[1]
                    and (rect_u[0] - rect_l[0]) > 0
                    and (rect_u[1] - rect_l[1]) > 0
                    and (not self.rectangle_comprises_other_upper_corner(rect_l, rect_u, self.upper_corners))
                ):
                    rectangles.append((rect_l, rect_u))
                    max_min_dists.append(self.max_min_distance_rect(rect_l, rect_u))
        z_l, z_u = rectangles[np.argmax(max_min_dists)]

        candidates = np.unique(
            np.stack(
                [(z_u + z_l) / 2, np.array([z_u[0], (z_u[1] + z_l[1]) / 2]), np.array([(z_u[0] + z_l[0]) / 2, z_u[1]])],
                axis=0,
            ),
            axis=0,
        )
        expansion_scenario_dist = self.compute_expansion_scenarios(z_l, z_u, candidates)
        best_candidate_idx = np.argmin(np.max(expansion_scenario_dist, axis=-1))
        # worst_case_dist_best_candidate = np.min(np.max(expansion_scenario_dist, axis=-1))
        z_next = candidates[best_candidate_idx]

        assert z_next.shape == (self.ndim,)
        return z_next

    def get_largest_minmax_rectangles(self):
        rectangles = []
        rectangle_diags = []
        upper_corners = self.upper_corners
        lower_corners = self.lower_corners_w_domain_corners
        for ru in upper_corners:
            for rl in np.concatenate([lower_corners, upper_corners]):
                print(ru, rl)
                rect_u, rect_l = copy.deepcopy(ru), copy.deepcopy(rl)
                if not (rect_l[0] <= rect_u[0]):
                    c = rect_l[0]
                    rect_l[0] = rect_u[0]
                    rect_u[0] = c
                if not (rect_l[1] <= rect_u[1]):
                    c = rect_l[1]
                    rect_l[1] = rect_u[1]
                    rect_u[1] = c
                if (
                    self.rectangle_is_between_frontiers(rect_u, rect_l)
                    and (rect_u[0] - rect_l[0]) > 0
                    and (rect_u[1] - rect_l[1]) > 0
                    and (not self.rectangle_comprises_other_upper_corner(rect_l, rect_u, upper_corners))
                ):
                    if not np.any([np.allclose(rect_l, rl) and np.allclose(rect_u, ru) for rl, ru in rectangles]):
                        rectangles.append((rect_l, rect_u))
                        rectangle_diags.append(np.sum((rect_u - rect_l) ** 2))

        # remove rectangles that are dominated by others
        idx = 0
        while idx < len(rectangles):
            rect_l, rect_u = rectangles[idx]
            is_dominated = np.sum([np.all(l <= rect_l) and np.all(rect_u <= u) for l, u in rectangles]) > 1
            if is_dominated:
                rectangles.pop(idx)
            else:
                idx += 1
        return rectangles

    def rectangle_comprises_other_upper_corner(self, rect_l, rect_u, upper_corners):
        a = np.any(
            np.logical_and(
                np.logical_and(rect_l[0] < upper_corners[:, 0], upper_corners[:, 0] < rect_u[0]),
                rect_u[1] == upper_corners[:, 1],
            )
        )
        b = np.any(
            np.logical_and(
                np.logical_and(rect_l[1] < upper_corners[:, 1], upper_corners[:, 1] < rect_u[1]),
                rect_u[0] == upper_corners[:, 0],
            )
        )
        return a or b

    @staticmethod
    def max_min_distance(lower_corners, upper_corners):
        dist_mat = scipy.spatial.distance_matrix(lower_corners, upper_corners)
        indices_min = np.argmin(dist_mat, axis=-1)
        min_dist_per_lower_corner = np.choose(indices_min, dist_mat.T)
        max_indices = np.arange(dist_mat.shape[0])[min_dist_per_lower_corner >= np.max(min_dist_per_lower_corner)]
        max_min_indices = [(idx_lower, indices_min[idx_lower]) for idx_lower in max_indices]
        dist = min_dist_per_lower_corner[max_min_indices[0][0]]
        return max_min_indices, dist

    @staticmethod
    def max_min_distance_fluid(lower_corners, upper_corners):
        dist_to_upper_boundary_per_l_corner = []
        for l_corner in lower_corners:
            up_corners_right = upper_corners[l_corner[0] <= upper_corners[:, 0]]
            if np.logical_and(
                np.any(l_corner[1] < up_corners_right[:, 1]), np.any(l_corner[1] > up_corners_right[:, 1])
            ):
                dist_to_bound_x = np.max(up_corners_right[:, 0] - l_corner[0])
            else:
                dist_to_bound_x = np.inf
            up_corners_top = upper_corners[l_corner[1] <= upper_corners[:, 1]]
            if np.logical_and(np.any(l_corner[0] < up_corners_top[:, 0]), np.any(l_corner[0] > up_corners_top[:, 0])):
                dist_to_bound_y = np.max(up_corners_top[:, 1] - l_corner[1])
            else:
                dist_to_bound_y = np.inf
            min_dist_to_normal_upper_corners = np.min(np.linalg.norm(upper_corners - l_corner, axis=-1))
            dist_to_upper_boundary_per_l_corner.append(
                min(min_dist_to_normal_upper_corners, dist_to_bound_x, dist_to_bound_y)
            )
        return max(dist_to_upper_boundary_per_l_corner)

    @property
    def max_min_distance_global(self):
        return self.max_min_distance_fluid(self.lower_corners_w_domain_corners, self.upper_corners)

    def _between_frontiers(self, point: np.ndarray):
        cond1 = np.all(
            np.logical_or(point[0] <= self.upper_frontier_points[:, 0], point[1] <= self.upper_frontier_points[:, 1])
        )
        cond2 = np.all(
            np.logical_or(point[0] >= self.lower_frontier_points[:, 0], point[1] >= self.lower_frontier_points[:, 1])
        )
        return all([cond1, cond2])

    def rectangle_is_between_frontiers(self, upper_corner: np.ndarray, lower_corner: np.ndarray) -> bool:
        # check that the rectangle with the provided upper and lower corner lies within the frontiers
        other_corner_1 = np.array([upper_corner[0], lower_corner[1]])
        other_corner_2 = np.array([lower_corner[0], upper_corner[1]])
        return all(
            [
                self._between_frontiers(upper_corner),
                self._between_frontiers(lower_corner),
                self._between_frontiers(other_corner_1),
                self._between_frontiers(other_corner_2),
            ]
        )

    @property
    def safe_evaluations(self) -> List[Tuple[np.ndarray, float, float]]:
        return [(point, constr, objective) for point, constr, objective in self.evaluations if constr >= 0]

    @property
    def unsafe_evaluations(self) -> List[Tuple[np.ndarray, float, float]]:
        return [(point, constr, objective) for point, constr, objective in self.evaluations if constr <= 0]

    def corner_points_upper(
        self,
        frontier_points: np.ndarray,
        lower_boundary: np.ndarray,
        upper_boundary: np.ndarray,
        add_boundary_points: bool = True,
    ) -> np.ndarray:
        if frontier_points.shape[0] > 1:
            upper_eval_points_sorted = frontier_points[frontier_points[:, 0].argsort()]
            outer_corner_points = np.stack([upper_eval_points_sorted[1:, 0], upper_eval_points_sorted[:-1, 1]], axis=-1)
            assert outer_corner_points.shape == (frontier_points.shape[0] - 1, 2)
            corner_points = np.concatenate([frontier_points, outer_corner_points], axis=0)
        else:
            corner_points = frontier_points
        if add_boundary_points:
            corner_points = np.concatenate(
                [corner_points, self.boundary_points_upper(frontier_points, lower_boundary, upper_boundary)], axis=0
            )
        return np.unique(corner_points, axis=0)

    def corner_points_lower(
        self,
        frontier_points: np.ndarray,
        lower_boundary: np.ndarray,
        upper_boundary: np.ndarray,
        add_boundary_points: bool = True,
    ) -> np.ndarray:
        if frontier_points.shape[0] > 1:
            frontier_points_sorted = frontier_points[frontier_points[:, 0].argsort()]
            outer_corner_points = np.stack([frontier_points_sorted[:-1, 0], frontier_points_sorted[1:, 1]], axis=-1)
            assert outer_corner_points.shape == (frontier_points.shape[0] - 1, 2)
            corner_points = np.concatenate([frontier_points, outer_corner_points], axis=0)
        else:
            corner_points = frontier_points
        if add_boundary_points:
            corner_points = np.concatenate(
                [corner_points, self.boundary_points_lower(frontier_points, lower_boundary, upper_boundary)], axis=0
            )
        return np.unique(corner_points, axis=0)

    def boundary_points_upper(
        self, frontier_points: np.ndarray, lower_boundary: np.ndarray, upper_boundary: np.ndarray
    ) -> np.ndarray:
        boundary_points = []
        for i in [0, 1]:
            min_i = np.min(frontier_points[:, i])
            if min_i > lower_boundary[i]:
                if i == 0:
                    boundary_points.append(np.array([min_i, upper_boundary[1]]))
                else:
                    boundary_points.append(np.array([upper_boundary[0], min_i]))
            else:
                boundary_points.append(frontier_points[np.argmin(frontier_points[:, i])])
        boundary_points = np.stack(boundary_points, axis=0)
        assert boundary_points.shape == (2, self.ndim)
        return boundary_points

    def boundary_points_lower(
        self, frontier_points: np.ndarray, lower_boundary: np.ndarray, upper_boundary: np.ndarray
    ) -> np.ndarray:
        boundary_points = []
        for i in [0, 1]:
            max_i = np.max(frontier_points[:, i])
            if max_i < upper_boundary[i]:
                if i == 0:
                    boundary_points.append(np.array([max_i, lower_boundary[1]]))
                else:
                    boundary_points.append(np.array([lower_boundary[0], max_i]))
            else:
                boundary_points.append(frontier_points[np.argmax(frontier_points[:, i])])
        boundary_points = np.stack(boundary_points, axis=0)
        assert boundary_points.shape == (2, self.ndim)
        return boundary_points

    @property
    def lower_corners_w_domain_corners(self) -> np.ndarray:
        lower_corners = self.corner_points_lower(
            self.lower_frontier_points, lower_boundary=self.lower_boundary, upper_boundary=self.upper_boundary
        )
        if np.all(self.upper_frontier_points[:, 0] > self.lower_boundary[0]) and (
            not np.any(lower_corners[:, 1] == self.upper_boundary[1])
        ):
            lower_corners = np.concatenate([lower_corners, self.upper_left_corner[None, :]], axis=0)
        if np.all(self.upper_frontier_points[:, 1] > self.lower_boundary[1]) and (
            not np.any(lower_corners[:, 0] == self.upper_boundary[0])
        ):
            lower_corners = np.concatenate([lower_corners, self.lower_right_corner[None, :]], axis=0)
        return lower_corners

    @property
    def lower_frontier_points_w_domain_corners(self):
        lower_front_points = self.lower_frontier_points
        if self.upper_frontier_points[:, 0] > self.lower_boundary[0]:
            lower_front_points = np.concatenate([lower_front_points, self.upper_left_corner[None, :]], axis=0)
        if self.upper_frontier_points[:, 1] > self.lower_boundary[1]:
            lower_front_points = np.concatenate([lower_front_points, self.lower_right_corner[None, :]], axis=0)
        return lower_front_points

    @property
    def upper_corners(self) -> np.ndarray:
        return self.corner_points_upper(
            frontier_points=self.upper_frontier_points,
            lower_boundary=self.lower_boundary,
            upper_boundary=self.upper_boundary,
            add_boundary_points=True,
        )

    def compute_expansion_scenarios(self, z_lower: np.ndarray, z_upper: np.ndarray, candidates: np.ndarray):
        assert candidates.ndim == 2 and candidates.shape[-1] == self.ndim
        upper_front_points = self.subset_for_rectangle(self.upper_corners, z_lower, z_upper, upper=True)
        lower_front_points = self.subset_for_rectangle(
            self.lower_corners_w_domain_corners, z_lower, z_upper, upper=False
        )
        upper_corners = self.corner_points_upper(upper_front_points, z_lower, z_upper)
        lower_corners = self.corner_points_lower(lower_front_points, z_lower, z_upper)

        max_min_dists = []
        for z_candidate in candidates:
            # lower case
            lower_front_w_candidate = np.concatenate([lower_corners, z_candidate[None, :]], axis=0)
            lower_front_w_candidate = self.prune_frontier_points(
                lower_front_w_candidate, lower_boundary=z_lower, upper_boundary=z_upper, upper=False, strict=True
            )
            lower_corners_w_candidate = self.corner_points_lower(lower_front_w_candidate, z_lower, z_upper)
            max_min_dist_lower_case = self.max_min_distance_fluid(lower_corners_w_candidate, upper_corners)

            # upper case
            upper_front_w_candidate = np.concatenate([upper_front_points, z_candidate[None, :]], axis=0)
            upper_front_w_candidate = self.prune_frontier_points(
                upper_front_w_candidate, lower_boundary=z_lower, upper_boundary=z_upper, upper=True, strict=True
            )
            upper_corners_w_candidate = self.corner_points_upper(upper_front_w_candidate, z_lower, z_upper)
            max_min_dist_upper_case = self.max_min_distance_fluid(lower_corners, upper_corners_w_candidate)

            max_min_dists.append((max_min_dist_lower_case, max_min_dist_upper_case))

        max_min_dists = np.array(max_min_dists)
        assert max_min_dists.shape == (len(candidates), 2)
        return max_min_dists

    def max_min_distance_rect(self, z_lower: np.ndarray, z_upper: np.ndarray):
        upper_front_points = self.subset_for_rectangle(self.upper_corners, z_lower, z_upper, upper=True)
        lower_front_points = self.subset_for_rectangle(
            self.lower_corners_w_domain_corners, z_lower, z_upper, upper=False
        )
        upper_corners = self.corner_points_upper(upper_front_points, z_lower, z_upper)
        lower_corners = self.corner_points_lower(lower_front_points, z_lower, z_upper)
        max_min_dist = self.max_min_distance_fluid(lower_corners, upper_corners)
        return max_min_dist

    def subset_for_rectangle(self, points: np.ndarray, z_lower: np.ndarray, z_upper: np.ndarray, upper: bool = True):
        within_rectangle_mask = np.all(np.logical_and(z_lower <= points, points <= z_upper), axis=-1)
        points_within_rectangle = points[within_rectangle_mask]
        if upper:
            project_points_mask = np.logical_or(
                np.logical_and(points[:, 0] < z_lower[0], points[:, 1] == z_upper[1]),
                np.logical_and(points[:, 1] < z_lower[1], points[:, 0] == z_upper[0]),
            )
        else:
            project_points_mask = np.logical_or(
                np.logical_and(points[:, 0] > z_upper[0], points[:, 1] == z_lower[1]),
                np.logical_and(points[:, 1] > z_upper[1], points[:, 0] == z_lower[0]),
            )
        projected_points = np.clip(points[project_points_mask], z_lower, z_upper)
        points_within_rectangle = np.concatenate([points_within_rectangle, projected_points], axis=0)
        points_within_rectangle = np.unique(points_within_rectangle, axis=0)
        assert points.shape[0] >= points_within_rectangle.shape[0]
        assert points.shape[1] == points_within_rectangle.shape[1]
        assert np.all(np.logical_and(z_lower <= points_within_rectangle, points_within_rectangle <= z_upper))
        return points_within_rectangle

    def upper_frontier(self, z: float, query_dim: int = 0) -> float:
        assert query_dim in [0, 1]
        relevant_points = self.upper_frontier_points[self.upper_frontier_points[:, query_dim] <= z]
        if relevant_points.shape[0] == 0:
            return self.upper_boundary[int(not query_dim)]
        else:
            return np.min(relevant_points[:, int(not query_dim)])

    def lower_frontier(self, z: float, query_dim: int = 0) -> float:
        assert query_dim in [0, 1]
        relevant_points = self.lower_frontier_points[self.lower_frontier_points[:, query_dim] >= z]
        if relevant_points.shape[0] == 0:
            return self.lower_boundary[int(not query_dim)]
        else:
            return np.max(relevant_points[:, int(not query_dim)])

    @property
    def num_evals(self) -> int:
        return len(self.evaluations)

    @property
    def best_safe_evaluation(self) -> Tuple[np.ndarray, float, float]:
        safe_evals = self.safe_evaluations
        best_idx = np.argmin([obj for _, _, obj in safe_evals])
        return safe_evals[best_idx]

    def add_eval(self, point: np.ndarray, constr: float, objective: float) -> None:
        self.evaluations.append((point, constr, objective))
        if constr >= 0.0:
            self.safe_frontier.add(point)
            self.upper_frontier_points = self.prune_frontier_points(
                np.concatenate([self.upper_frontier_points, point[None, :]]),
                lower_boundary=self.lower_boundary,
                upper_boundary=self.upper_boundary,
                upper=True,
            )
        else:
            self.unsafe_frontier.add(point)
            self.lower_frontier_points = self.prune_frontier_points(
                np.concatenate([self.lower_frontier_points, point[None, :]]),
                lower_boundary=self.lower_boundary,
                upper_boundary=self.upper_boundary,
                upper=False,
            )

    @staticmethod
    def prune_frontier_points(
        frontier_points: np.ndarray,
        lower_boundary: np.ndarray,
        upper_boundary: np.ndarray,
        upper: bool = True,
        strict: bool = False,
    ) -> np.ndarray:
        def is_dominated(a, b):
            if upper:
                return (np.asarray(a) > b).all() if strict else (np.asarray(a) >= b).all()
            else:
                return (np.asarray(a) < b).all() if strict else (np.asarray(a) <= b).all()

        frontier_points = np.unique(frontier_points, axis=0)
        dominated_mat = scipy.spatial.distance.cdist(frontier_points, frontier_points, metric=is_dominated)
        np.fill_diagonal(dominated_mat, np.zeros(dominated_mat.shape[0]))
        point_not_dominated = np.sum(dominated_mat, axis=-1) == 0.0
        pruned_frontier_points = frontier_points[point_not_dominated]

        if not upper:
            # if lower frontier, remove points that lie on the upper boundary and are weakly dominated
            weakly_dominated_mat = scipy.spatial.distance.cdist(
                pruned_frontier_points, pruned_frontier_points, metric=lambda a, b: (a <= b).all()
            )
            np.fill_diagonal(weakly_dominated_mat, np.zeros(weakly_dominated_mat.shape[0]))
            point_weakly_dominated = np.any(weakly_dominated_mat, axis=-1)
            point_on_upper_boundary = np.logical_or(
                pruned_frontier_points[:, 0] == upper_boundary[0], pruned_frontier_points[:, 1] == upper_boundary[1]
            )
            pruned_frontier_points = pruned_frontier_points[
                np.logical_not(np.logical_and(point_weakly_dominated, point_on_upper_boundary))
            ]
        return pruned_frontier_points

    @staticmethod
    def prune_corner_points(corner_points: np.ndarray, upper: bool = True):
        def not_dominated(a, b):
            if upper:
                return (np.asarray(a) <= b).all()
            else:
                return (np.asarray(a) >= b).all()

        corner_points = np.unique(corner_points, axis=0)
        not_dominated_mat = scipy.spatial.distance.cdist(corner_points, corner_points, metric=not_dominated)
        np.fill_diagonal(not_dominated_mat, np.ones(not_dominated_mat.shape[0]))
        point_not_dominated = np.any(not_dominated_mat, axis=-1)
        return corner_points[point_not_dominated]


if __name__ == "__main__":
    # setup constr function
    x1, x2 = np.meshgrid(np.arange(-5, 3, 0.05), np.arange(-5, 3, 0.05))

    def q(x):
        if x.ndim == 1:
            x = x.reshape((1, -2))
        return 1.0 * x[:, 0] + x[:, 1] ** 3

    def loss(x):
        if x.ndim == 1:
            x = x.reshape((1, -2))
        return np.linalg.norm(x - np.ones(ndim), axis=-1)

    y = q(np.stack([x1.flatten(), x2.flatten()], axis=-1)).reshape(x1.shape)

    lower = np.array([-5.0, -5.0])
    upper = np.array([3.0, 3.0])
    ndim = 2
    optim = MonotoneFrontierSolverV2(ndim=ndim, lower_boundary=lower, upper_boundary=upper)

    for i in range(71):
        x = optim.next()
        print(i, optim.max_min_distance_global, x)
        constr = q(x)
        objective = 1.0
        optim.add_eval(point=x, constr=constr, objective=loss(x))

        if i % 1 == 0:

            def plot_sets():
                plt.contour(x1, x2, y, np.array([0.0, 100.0]), origin="lower")
                plt.colorbar()

                for corner_point in optim.lower_frontier_points:
                    x_set = np.linspace(lower[0], corner_point[0], 100)
                    plt.fill_between(
                        x_set,
                        lower[1] * np.ones_like(x_set),
                        corner_point[1] * np.ones_like(x_set),
                        color="red",
                        alpha=0.3,
                    )

                lower_corner_points = optim.lower_corners_w_domain_corners
                plt.scatter(lower_corner_points[:, 0], lower_corner_points[:, 1], color="red")

                x_front = np.linspace(lower[0], upper[0], 200)
                front = [optim.upper_frontier(x, 0) for x in x_front]
                plt.plot(x_front, front, color="green")

                corner_points_upper = optim.upper_corners
                plt.scatter(corner_points_upper[:, 0], corner_points_upper[:, 1], color="green")

                for corner_point in optim.upper_frontier_points:
                    x_set = np.linspace(corner_point[0], upper[0], 100)
                    plt.fill_between(
                        x_set,
                        corner_point[1] * np.ones_like(x_set),
                        upper[1] * np.ones_like(x_set),
                        color="green",
                        alpha=0.3,
                    )

                x_front = np.linspace(lower[0], upper[0], 200)
                front = [optim.lower_frontier(x, 0) for x in x_front]
                plt.plot(x_front, front, color="red")

                x_best = optim.best_safe_evaluation[0]
                plt.scatter(x_best[0], x_best[1])
                # plt.scatter(np.stack(optim.candidate_points, axis=0)[:, 0], np.stack(optim.candidate_points, axis=0)[:, 1])
                plt.title(f"iter {i}")
                plt.show()

            plot_sets()

        plt.show()
