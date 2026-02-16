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
from alef.pools.base_pool import BasePool


class PoolFromOracle(BasePool):
    def __init__(self, oracle: BaseOracle, seed: int = 123, set_seed: bool = False):
        super().__init__()
        self.oracle = oracle
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
            X, Y = self.get_random_data(int(100**D), noisy=False)
            return max(Y)

    def get_constrained_max(self, constraint_lower: float = -np.inf, constraint_upper: float = np.inf):
        max_unconst = self.get_max()
        assert max_unconst >= constraint_lower
        return min(max_unconst, constraint_upper)

    def query(self, x, noisy: bool):
        if np.shape(np.atleast_2d(x))[0] > 1:
            raise ValueError("please query only 1 point")

        idx = row_wise_compare(self.__x, x)
        idx = np.where(idx)[0]
        if len(idx) < 1 and not self._query_non_exist_points:
            raise ValueError("queried point does not exist")

        y = self.oracle.query(x, noisy)

        if not self._with_replacement:
            self.__x = np.delete(self.__x, (idx), axis=0)

        return y

    def get_grid_data(self, *args, **kwargs):
        """
        input: n_per_dim noisy
        """
        if hasattr(self.oracle, "get_grid_data"):
            return self.oracle.get_grid_data(*args, **kwargs)
        else:
            raise NotImplementedError(f"{self.oracle.__class__.__name__} does not have 'get_grid_data' method")

    def load_grid_data(self, folder: str):
        X_path = os.path.join(folder, "X_grid.txt")
        Y_path = os.path.join(folder, "Y_grid.txt")
        return self.load_data(X_path, Y_path)

    def get_random_data(self, *args, **kwargs):
        """
        input: n, noisy
        """
        return self.oracle.get_random_data(*args, **kwargs)

    def get_random_data_in_box(
        self, n: int, a: Union[float, Sequence[float]], box_width: Union[float, Sequence[float]], noisy: bool = True
    ):
        return self.oracle.get_random_data_in_box(n, a, box_width, noisy)

    def get_random_constrained_data(
        self, n: int, noisy: bool = True, constraint_lower: float = -np.inf, constraint_upper: float = np.inf
    ):
        n_constr = 0
        X, Y = None, None

        while n_constr < n:
            X_rd, Y_rd = self.get_random_data(n, noisy=noisy)
            X, Y = self._constrain_data(X, Y, X_rd, Y_rd, n, constraint_lower, constraint_upper)
            n_constr = X.shape[0]

        return X, Y

    def get_random_constrained_data_in_box(
        self,
        n: int,
        a: Union[float, Sequence[float]],
        box_width: Union[float, Sequence[float]],
        noisy: bool = True,
        constraint_lower: float = -np.inf,
        constraint_upper: float = np.inf,
    ):
        n_constr = 0
        X, Y = None, None

        while n_constr < n:
            X_rd, Y_rd = self.get_random_data_in_box(n, a, box_width, noisy=noisy)
            X, Y = self._constrain_data(X, Y, X_rd, Y_rd, n, constraint_lower, constraint_upper)
            n_constr = X.shape[0]

        return X, Y

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

    def _constrain_data(
        self, X_origin, Y_origin, X_new, Y_new, n_target: int, constraint_lower: float, constraint_upper: float
    ):
        r"""
        return X, Y
            X are elements of X_origin and X_new that have corresponding Y under the constraint
            Y are corresponding to X
        """
        mask = np.logical_and(Y_new >= constraint_lower, Y_new <= constraint_upper)[:, 0]

        if X_origin is None:
            X = X_new[mask]
            Y = Y_new[mask]
        else:
            X = np.vstack((X_origin, X_new[mask]))
            Y = np.vstack((Y_origin, Y_new[mask]))

        X, unique_idx = row_wise_unique(X)
        Y = Y[unique_idx]

        n_constr = X.shape[0]
        idx = np.random.choice(n_constr, size=min(n_target, n_constr), replace=False)

        return X[idx], Y[idx]


if __name__ == "__main__":
    PoolFromOracle(None).get_grid_data(10, False)
