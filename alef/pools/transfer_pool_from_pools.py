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
from scipy import stats
from alef.enums.data_structure_enums import OutputType
from alef.enums.multi_task_enums import TransferLearningTaskMode as TaskMode
from alef.pools.base_pool import BasePool


class TransferPoolFromPools(BasePool):
    """
    This pool take one source_pool and one target_pool.
    It won't modify the data in each pool, but whenever data are retrieve or processed,
        it will decorate them in the flattened multi_output way (i.e. X: [N, D+1] array, with the last column being output indices)
    """

    def __init__(self, source_pool, target_pool):
        super().__init__()
        self.output_type = OutputType.MULTI_OUTPUT_FLATTENED

        self.source_pool = source_pool
        self.target_pool = target_pool

        self.task_mode = TaskMode.SOURCE
        self.source_dimension = self.source_pool.get_dimension()
        self.target_dimension = self.target_pool.get_dimension()

        # self.source_pool.output_type = OutputType.MULTI_OUTPUT_FLATTENED
        # self.target_pool.output_type = OutputType.MULTI_OUTPUT_FLATTENED

    def set_query_non_exist(self, query_non_exist_points: bool):
        super().set_query_non_exist(query_non_exist_points)
        self.source_pool.set_query_non_exist(query_non_exist_points)
        self.source_pool.set_query_non_exist(query_non_exist_points)

    def set_task_mode(self, learning_target: bool = False):
        if learning_target:
            self.task_mode = TaskMode.TARGET
        else:
            self.task_mode = TaskMode.SOURCE

    @property
    def task_index(self):
        return int(self.task_mode.value)

    def data_decorator(self, x, D: int, p: int):
        r"""
        x: [D,] array, [N, D] array or float (treat as [1, 1] array)
        return: [N, D+1] array, the last column is p
        """
        xx = np.atleast_2d(x)[..., :D]
        N = xx.shape[0]

        return np.hstack((xx, np.ones([N, 1]) * p))

    def data_tuple_decorator(self, data, D: int, p: int):
        r"""
        data: (x, y) or (x, y, z)
            x: [D,] array, [N, D] array or float (treat as [1, 1] array)
            y: [1,], [N, 1], or float
            z: [Q,], [N, Q], or float, where Q is the number of safety controls

        decorate x into x': [N, D+1] array, the last column is p

        return: (x', y, z)
        """
        X = self.data_decorator(data[0], D, p)
        YZ_list = data[1:]
        return (X, *YZ_list)

    def _get_pool_d_p(self, task_mode):
        if task_mode == TaskMode.SOURCE:
            pool = self.source_pool
            d = self.source_dimension
            p = TaskMode.SOURCE.value
        elif task_mode == TaskMode.TARGET:
            pool = self.target_pool
            d = self.target_dimension
            p = TaskMode.TARGET.value
        else:
            return ValueError("Unknown task_mode")
        return pool, d, p

    def get_max(self):
        pool, _, _ = self._get_pool_d_p(self.task_mode)
        if hasattr(pool, "get_max"):
            return pool.get_max()
        else:
            raise NotImplementedError

    def get_constrained_max(self, constraint_lower: float = -np.inf, constraint_upper: float = np.inf):
        pool, _, _ = self._get_pool_d_p(self.task_mode)
        if hasattr(pool, "get_constrained_max"):
            return pool.get_constrained_max(constraint_lower, constraint_upper)
        else:
            raise NotImplementedError

    def query(self, x, noisy=True):
        pool, d, p = self._get_pool_d_p(self.task_mode)
        return pool.query(x[..., :d], noisy=noisy)

    def get_grid_data(self, *args, **kwargs):
        pool, d, p = self._get_pool_d_p(self.task_mode)
        # output_tuple may be (X, Y) or (X, Y, Z)
        output_tuple = pool.get_grid_data(*args, **kwargs)
        return self.data_tuple_decorator(output_tuple, d, p)

    def load_grid_data(self, *args, **kwargs):
        pool, d, p = self._get_pool_d_p(self.task_mode)
        # output_tuple may be (X, Y) or (X, Y, Z)
        output_tuple = pool.load_grid_data(*args, **kwargs)
        return self.data_tuple_decorator(output_tuple, d, p)

    def get_data_from_idx(self, *args, **kwargs):
        pool, d, p = self._get_pool_d_p(self.task_mode)
        # output_tuple may be (X, Y) or (X, Y, Z)
        output_tuple = pool.get_data_from_idx(*args, **kwargs)
        return self.data_tuple_decorator(output_tuple, d, p)

    def get_random_data(self, *args, **kwargs):
        pool, d, p = self._get_pool_d_p(self.task_mode)
        # output_tuple may be (X, Y) or (X, Y, Z)
        output_tuple = pool.get_random_data(*args, **kwargs)
        return self.data_tuple_decorator(output_tuple, d, p)

    def get_random_data_in_box(self, *args, **kwargs):
        pool, d, p = self._get_pool_d_p(self.task_mode)
        # output_tuple may be (X, Y) or (X, Y, Z)
        output_tuple = pool.get_random_data_in_box(*args, **kwargs)
        return self.data_tuple_decorator(output_tuple, d, p)

    def get_random_constrained_data(self, *args, **kwargs):
        pool, d, p = self._get_pool_d_p(self.task_mode)
        # output_tuple may be (X, Y) or (X, Y, Z)
        output_tuple = pool.get_random_constrained_data(*args, **kwargs)
        return self.data_tuple_decorator(output_tuple, d, p)

    def get_random_constrained_data_in_box(self, *args, **kwargs):
        pool, d, p = self._get_pool_d_p(self.task_mode)
        # output_tuple may be (X, Y) or (X, Y, Z)
        output_tuple = pool.get_random_constrained_data_in_box(*args, **kwargs)
        return self.data_tuple_decorator(output_tuple, d, p)

    def get_dimension(self, *args, **kwargs):
        if self.task_mode == TaskMode.SOURCE:
            return self.source_dimension
        elif self.task_mode == TaskMode.TARGET:
            return self.target_dimension
        else:
            return ValueError("Unknown task_mode")

    def get_variable_dimension(self, *args, **kwargs):
        pool, d, p = self._get_pool_d_p(self.task_mode)
        return pool.get_variable_dimension(*args, **kwargs)

    def set_replacement(self, with_replacement: bool):
        if self.task_mode == TaskMode.SOURCE:
            return self.source_pool.set_replacement(with_replacement=with_replacement)
        elif self.task_mode == TaskMode.TARGET:
            return self.target_pool.set_replacement(with_replacement=with_replacement)
        else:
            return ValueError("Unknown task_mode")

    def get_context_status(self, *args, **kwargs):
        pool, d, p = self._get_pool_d_p(self.task_mode)
        if hasattr(pool, "get_context_status"):
            return pool.get_context_status(*args, **kwargs)
        else:
            raise NotImplementedError

    @property
    def __x(self):
        pool, d, p = self._get_pool_d_p(self.task_mode)
        return self.data_decorator(pool.possible_queries(), d, p)

    def possible_queries(self):
        return self.__x

    def get_pearson_correlation(self, noisy: bool = False):
        r"""
        this is actually a bad method in general
        only use this when source and target pools are from oracles
        """
        n = 5000
        output_tuple = self.source_pool.get_random_data(n, noisy)

        X = output_tuple[0]
        Ys = output_tuple[1]
        Yt = np.empty_like(Ys)
        if len(output_tuple) == 3:
            Zs = output_tuple[2]
            Zt = np.empty_like(Zs)

        replacement = self.target_pool._with_replacement
        query_nonexist = self.target_pool._query_non_exist_points
        self.target_pool.set_replacement(True)
        self.target_pool.set_query_non_exist(True)
        for i in range(n):
            if len(output_tuple) == 2:
                Yt[i, :] = self.target_pool.query(X[i, :], noisy)
            elif len(output_tuple) == 3:
                Yt[i, :], Zt[i, :] = self.target_pool.query(X[i, :], noisy)
        self.target_pool.set_replacement(replacement)
        self.target_pool.set_query_non_exist(query_nonexist)

        if len(output_tuple) == 2:
            cr_y, _ = stats.pearsonr(Ys.reshape(-1), Yt.reshape(-1))
            return cr_y
        elif len(output_tuple) == 3:
            cr_y, _ = stats.pearsonr(Ys.reshape(-1), Yt.reshape(-1))
            cr_z, _ = stats.pearsonr(Zs.reshape(-1), Zt.reshape(-1))
            return cr_y, cr_z
