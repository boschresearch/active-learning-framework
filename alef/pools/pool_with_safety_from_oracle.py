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

import os
from typing import Union, Sequence
import numpy as np
from alef.utils.utils import row_wise_compare, row_wise_unique
from alef.oracles.base_oracle import BaseOracle
from alef.pools.base_pool_with_safety import BasePoolWithSafety


class PoolWithSafetyFromOracle(BasePoolWithSafety):
    def __init__(
        self,
        oracle: BaseOracle,
        safety_oracle: Union[BaseOracle, Sequence[BaseOracle]] = None,
        seed: int = 123,
        set_seed: bool = False,
    ):
        super().__init__()

        self.oracle = oracle
        if isinstance(safety_oracle, list):
            self.safety_oracle = safety_oracle
        else:
            self.safety_oracle = [safety_oracle]
        self.__x = None
        if set_seed:
            np.random.seed(seed)

    def get_max(self):
        if hasattr(self.oracle, "get_max"):
            return self.oracle.get_max()
        else:
            D = self.get_dimension()
            if D > 3:
                print(f"dimension is {D}, this might take a while")
            X, Y, Z = self.get_random_data(int(100**D), noisy=False)
            return max(Y)

    def get_constrained_max(self, constraint_lower: float = -np.inf, constraint_upper: float = np.inf):
        D = self.get_variable_dimension()
        if D > 3:
            print(f"dimension is {D}, this might take a while")
        X, Y, Z = self.get_random_constrained_data(
            int(100**D), noisy=False, constraint_lower=constraint_lower, constraint_upper=constraint_upper
        )
        return max(Y)

    def query(self, x, noisy=True):
        if np.shape(np.atleast_2d(x))[0] > 1:
            raise ValueError("please query only 1 point")

        idx = row_wise_compare(self.__x, x)
        idx = np.where(idx)[0]
        if len(idx) < 1 and not self._query_non_exist_points:
            raise ValueError("queried point does not exist")

        y = self.oracle.query(x, noisy)
        z = []
        for oracle in self.safety_oracle:
            z.append(oracle.query(x, noisy))
        z = np.array(z)

        if not self._with_replacement:
            self.__x = np.delete(self.__x, (idx), axis=0)

        return y, z

    def get_grid_data(self, n_per_dim: int, noisy: bool = True):
        """
        input: n_per_dim noisy
        """
        if not hasattr(self.oracle, "get_grid_data"):
            raise NotImplementedError(f"{self.oracle.__class__.__name__} does not have 'get_grid_data' method")
        if np.any([not hasattr(so, "get_grid_data") for so in self.safety_oracle]):
            raise NotImplementedError("At least one safety_oracle does not have 'get_grid_data' method")

        X, Y = self.oracle.get_grid_data(n_per_dim, noisy)
        Z = np.empty([X.shape[0], len(self.safety_oracle)])
        for i in range(X.shape[0]):
            for j in range(len(self.safety_oracle)):
                Z[i, j] = self.safety_oracle[j].query(X[i, :], noisy)
        return X, Y, Z

    def load_grid_data(self, folder: str):
        X_path = os.path.join(folder, "X_grid.txt")
        Y_path = os.path.join(folder, "Y_grid.txt")
        Z_path = os.path.join(folder, "Z_grid.txt")
        return self.load_data(X_path, Y_path, Z_path)

    def get_random_data(self, n, noisy=True):
        X, Y = self.oracle.get_random_data(n, noisy)
        Z = np.empty([X.shape[0], len(self.safety_oracle)])
        for i in range(X.shape[0]):
            for j in range(len(self.safety_oracle)):
                Z[i, j] = self.safety_oracle[j].query(X[i, :], noisy)
        return X, Y, Z

    def get_random_data_in_box(
        self, n: int, a: Union[float, Sequence[float]], box_width: Union[float, Sequence[float]], noisy: bool = True
    ):
        X, Y = self.oracle.get_random_data_in_box(n, a, box_width, noisy)
        Z = np.empty([X.shape[0], len(self.safety_oracle)])
        for i in range(X.shape[0]):
            for j in range(len(self.safety_oracle)):
                Z[i, j] = self.safety_oracle[j].query(X[i, :], noisy)
        return X, Y, Z

    def get_random_constrained_data(
        self,
        n: int,
        noisy: bool = True,
        constraint_lower: Union[float, Sequence[float]] = -np.inf,
        constraint_upper: Union[float, Sequence[float]] = np.inf,
    ):
        bound_low, bound_upp = self._assert_constraint(constraint_lower, constraint_upper)

        n_constr = 0
        X, Y, Z = None, None, None

        while n_constr < n:
            X_rd, Y_rd, Z_rd = self.get_random_data(n, noisy)
            X, Y, Z = self._constrain_data(X, Y, Z, X_rd, Y_rd, Z_rd, n, bound_low, bound_upp)
            n_constr = X.shape[0]

        return X, Y, Z

    def get_random_constrained_data_in_box(
        self,
        n: int,
        a: Union[float, Sequence[float]],
        box_width: Union[float, Sequence[float]],
        noisy: bool = True,
        constraint_lower: Union[float, Sequence[float]] = -np.inf,
        constraint_upper: Union[float, Sequence[float]] = np.inf,
    ):
        """
        args: a, b, box_width
        or:   a, box_width
        """
        bound_low, bound_upp = self._assert_constraint(constraint_lower, constraint_upper)

        n_constr = 0
        X, Y, Z = None, None, None

        while n_constr < n:
            X_rd, Y_rd, Z_rd = self.get_random_data_in_box(n, a, box_width, noisy=noisy)
            X, Y, Z = self._constrain_data(X, Y, Z, X_rd, Y_rd, Z_rd, n, bound_low, bound_upp)
            n_constr = X.shape[0]

        return X, Y, Z

    def get_box_bounds(self, *args, **kwargs):
        return self.oracle.get_box_bounds(*args, **kwargs)

    def get_dimension(self, *args, **kwargs):
        return self.oracle.get_dimension(*args, **kwargs)

    def set_data(self, x_data: np.ndarray):
        r"""
        set x manually
        """
        self.__x = x_data.copy()

    def discretize_random(self, n: int):
        r"""
        set x randomly from the space defined in the oracle (get discretized input space from the oracle)
        """
        self.__x = np.random.uniform(*self.get_box_bounds(), size=(n, self.get_dimension()))

    def possible_queries(self):
        return self.__x.copy()

    def get_context_status(self, *args, **kwargs):
        return self.oracle.get_context_status(*args, **kwargs)

    def _assert_constraint(self, constraint_lower, constraint_upper):
        """
        format check,
        return constraint_lower, constraint_upper
            after applying decoration
        """
        if not hasattr(constraint_lower, "__len__"):
            bound_low = [constraint_lower] * len(self.safety_oracle)
        else:
            bound_low = constraint_lower
            assert len(bound_low) == len(self.safety_oracle)

        if not hasattr(constraint_upper, "__len__"):
            bound_upp = [constraint_upper] * len(self.safety_oracle)
        else:
            bound_upp = constraint_upper
            assert len(bound_upp) == len(self.safety_oracle)

        return bound_low, bound_upp

    def _constrain_data(
        self,
        X_origin,
        Y_origin,
        Z_origin,
        X_new,
        Y_new,
        Z_new,
        n_target: int,
        constraint_lower: Union[float, Sequence[float]],
        constraint_upper: Union[float, Sequence[float]],
    ):
        r"""
        return X, Y
            X are elements of X_origin and X_new that have corresponding Y under the constraint
            Y are corresponding to X
        """
        mask = np.logical_and(Z_new >= constraint_lower, Z_new <= constraint_upper)
        mask = np.all(mask, axis=-1)

        if X_origin is None:
            X = X_new[mask]
            Y = Y_new[mask]
            Z = Z_new[mask]
        else:
            X = np.vstack((X_origin, X_new[mask]))
            Y = np.vstack((Y_origin, Y_new[mask]))
            Z = np.vstack((Z_origin, Z_new[mask]))

        X, unique_idx = row_wise_unique(X)
        Y = Y[unique_idx]
        Z = Z[unique_idx]

        n_constr = X.shape[0]
        idx = np.random.choice(n_constr, size=min(n_target, n_constr), replace=False)

        return np.atleast_2d(X[idx]), np.atleast_2d(Y[idx]), np.atleast_2d(Z[idx])
