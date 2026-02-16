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
from typing import Tuple, Union, Sequence
from abc import ABC, abstractmethod
from alef.enums.data_structure_enums import OutputType


class BasePool(ABC):
    def __init__(self):
        super().__init__()
        self.output_type = OutputType.SINGLE_OUTPUT
        self._with_replacement = False
        self._query_non_exist_points = False

    def get_replacement(self):
        return self._with_replacement

    def set_replacement(self, with_replacement: bool):
        self._with_replacement = with_replacement

    def get_query_non_exist(self):
        return self._query_non_exist_points

    def set_query_non_exist(self, query_non_exist_points: bool):
        self._query_non_exist_points = query_non_exist_points

    def load_data(self, X_path: str, Y_path: str):
        X = np.loadtxt(X_path)
        Y = np.loadtxt(Y_path)
        return np.atleast_2d(X), np.atleast_2d(Y)

    @abstractmethod
    def query(self, x: np.ndarray) -> np.float:
        """
        Queries the pool at location x and gets back the value

        Arguments:
            x : np.array - np.arry with dimension (d,) where d is the input dimension
        Returns:
            np.float - value of pool at location x
        """
        raise NotImplementedError

    @abstractmethod
    def get_random_data(self, n: int, noisy: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates random n uniform random queries inside the box bounds of the pool and return the queries and pool values

        Arguments:
            n : int - number of uniform random queries
            noisy : bool - flag if noise should be added - should raise Error if noisy=False is not possible if e.g. real world pool without ground truth
        Returns:
            np.array - input/queries with dimension (n,d) where d is the input dimension
            np.array - pool values at input dimension = (n,1)
        """
        raise NotImplementedError

    @abstractmethod
    def get_random_data_in_box(
        self, n: int, a: Union[float, Sequence[float]], box_width: Union[float, Sequence[float]], noisy: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates random n uniform random queries inside the specified box bounds and return the queries and oracle values

        Arguments:
            n : int - number of uniform random queries
            a: Union[float, Sequence[float]] - lower bound(s) of the specified box
            box_width: Union[float, Sequence[float]] - box width(s) of the specified box, so the box will be from a to a+box_width
            noisy : bool - flag if noise should be added - should raise Error if noisy=False is not possible if e.g. real world oracle without ground truth
        Returns:
            np.array - input/queries with dimension (n,d) where d is the input dimension
            np.array - oracle values at input dimension = (n,1)
        """
        raise NotImplementedError

    @abstractmethod
    def get_dimension(self):
        """
        Returns input dimension of pool

        Returns:
            int - input dimension
        """
        raise NotImplementedError

    def get_variable_dimension(self):
        """
        Returns variable dimension of input in the pool
        This may be less than input dimension when we fixed few inputs to some values

        Returns:
            int - input dimension
        """
        return self.get_dimension()

    @abstractmethod
    def possible_queries(self):
        """
        return possible inputs
        """
        raise NotImplementedError
