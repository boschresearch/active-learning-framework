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
import os
from typing import Tuple, Union, Sequence
from matplotlib import pyplot as plt
from alef.utils.utils import create_grid, check1Dlist
from alef.oracles.context_interface import ContextSupport


class BaseOracle:
    def query(self, x: np.array) -> float:
        """
        Queries the oracle at location x and gets back the oracle value

        Arguments:
            x : np.array - np.arry with dimension (d,) where d is the input dimension
        Returns:
            float - value of oracle at location x
        """
        raise NotImplementedError

    def get_random_data(self, n: int, noisy: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates random n uniform random queries inside the box bounds of the oracle and return the queries and oracle values

        Arguments:
            n : int - number of uniform random queries
            noisy : bool - flag if noise should be added - should raise Error if noisy=False is not possible if e.g. real world oracle without ground truth
        Returns:
            np.array - input/queries with dimension (n,d) where d is the input dimension
            np.array - oracle values at input dimension = (n,1)
        """
        raise NotImplementedError

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

    def get_box_bounds(self):
        """
        Returns box bounds a,b of the oracle (right now every dimension has the same bounds)

        Returns:
            float - lower box bound
            float - upper box bound
        """
        raise NotImplementedError

    def get_dimension(self):
        """
        Returns input dimension of oracle

        Returns:
            int - input dimension
        """
        raise NotImplementedError


class StandardOracle(BaseOracle, ContextSupport):
    def __init__(self, observation_noise: float, a: float, b: float, dimension: int):
        self.__a = a
        self.__b = b
        self.__dimension = dimension
        self.observation_noise = observation_noise

    def get_max(self):
        D = self.get_dimension()
        a, b = self.get_box_bounds()
        dx = 0.1
        n_per_dim = (b - a) / dx
        n = int(n_per_dim**D)

        X, Y = self.get_random_data(n, noisy=False)
        return max(Y)

    def get_grid_data(self, n_per_dim: int, noisy: bool = True):
        X = create_grid(self.__a, self.__b, n_per_dim, self.get_variable_dimension())
        X = self._decorate_variable_with_context(X)

        function_values = [np.reshape(self.query(x, noisy), [1, -1]) for x in X]
        return X, np.concatenate(function_values, axis=0)

    def get_random_data(self, n: int, noisy: bool = True):
        r"""
        Generates random n uniform random queries inside the box bounds of the oracle and return the queries and oracle values

        Arguments:
            n : int - number of uniform random queries
            noisy : bool - flag if noise should be added - should raise Error if noisy=False is not possible if e.g. real world oracle without ground truth
        Returns:
            np.array - input/queries with dimension (n,d) where d is the input dimension
            np.array - oracle values at input dimension = (n,1)
        """
        X = np.random.uniform(low=self.__a, high=self.__b, size=(n, self.get_variable_dimension()))
        X = self._decorate_variable_with_context(X)
        function_values = [np.reshape(self.query(x, noisy), [1, -1]) for x in X]
        return X, np.concatenate(function_values, axis=0)

    def get_random_data_in_box(
        self, n: int, a: Union[float, Sequence[float]], box_width: Union[float, Sequence[float]], noisy: bool = True
    ):
        r"""
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
        aa = np.array(check1Dlist(a, self.get_dimension()))
        bw = np.array(check1Dlist(box_width, self.get_dimension()))
        bb = aa + bw
        use_context, context_idx, context_values = self.get_context_status(return_idx=True, return_values=True)
        if use_context:
            aa[context_idx] = context_values
            bb[context_idx] = context_values

        assert np.all(aa <= self.__b)
        assert np.all(bb >= self.__a)
        assert np.any(aa < self.__b)
        assert np.any(bb > self.__a)

        aa[aa < self.__a] = self.__a
        bb[bb > self.__b] = self.__b

        X = np.random.uniform(low=aa, high=bb, size=(n, self.get_dimension()))

        function_values = [np.reshape(self.query(x, noisy), [1, -1]) for x in X]
        return X, np.concatenate(function_values, axis=0)

    def get_box_bounds(self):
        """
        Returns box bounds a,b of the oracle (right now every dimension has the same bounds)

        Returns:
            float - lower box bound
            float - upper box bound
        """
        return self.__a, self.__b

    def set_box_bounds(self, a: float, b: float):
        """
        Method for setting the box bounds manually

        Arguments:
            a : float - lower bound
            b : gloat - upper bound
        """
        self.__a = a
        self.__b = b

    def get_dimension(self):
        r"""
        Returns input dimension of oracle

        Returns:
            int - input dimension
        """
        return self.__dimension


class Standard1DOracle(StandardOracle):
    def __init__(self, observation_noise: float, a: float, b: float):
        super().__init__(observation_noise, a, b, 1)

    def plot(self):
        self._plot()
        plt.show()

    def save_plot(self, file_name: str = None, file_path: str = None):
        self._plot()

        plt.tight_layout()
        plt.savefig(os.path.join(file_path, file_name))
        plt.close()

    def _plot(self):
        xs, ys = self.get_random_data(500, True)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(xs, ys, marker=".", color="black")
        return fig


class Standard2DOracle(StandardOracle):
    def __init__(self, observation_noise: float, a: float, b: float):
        super().__init__(observation_noise, a, b, 2)

    def plot(self):
        self._plot_3D()
        plt.show()

    def save_plot(self, file_name: str = None, file_path: str = None):
        self._plot_3D()

        plt.tight_layout()
        plt.savefig(os.path.join(file_path, file_name))
        plt.close()

    def plot_color_map(self, safe_lower_bound: float = 1 / 6, safe_upper_bound: float = np.inf):
        self._plot_color_map(safe_lower_bound, safe_upper_bound)
        plt.show()

    def save_color_map(
        self,
        safe_lower_bound: float = 1 / 6,
        safe_upper_bound: float = np.inf,
        file_name: str = None,
        file_path: str = None,
    ):
        self._plot_color_map(safe_lower_bound, safe_upper_bound)

        plt.tight_layout()
        plt.savefig(os.path.join(file_path, file_name))
        plt.close()

    def _plot_3D(self):
        xs, ys = self.get_random_data(2000, True)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(xs[:, 0], xs[:, 1], ys, marker=".", color="black")

        return fig

    def _plot_color_map(self, safe_lower_bound: float = 1 / 6, safe_upper_bound: float = np.inf):
        xs, ys = self.get_random_data(2000, True)

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 2, 1)
        contour = ax.tricontourf(xs[:, 0], xs[:, 1], ys[:, 0], levels=np.linspace(-2, 2, 50), cmap="seismic")
        fig.colorbar(contour, ax=ax)
        ax = fig.add_subplot(1, 2, 2)
        contour = ax.tricontourf(
            xs[:, 0],
            xs[:, 1],
            (ys[:, 0] >= safe_lower_bound) * (ys[:, 0] <= safe_upper_bound),
            levels=[-0.5, 0.5, 1.5],
            cmap="YlGn",
        )
        fig.colorbar(contour, ax=ax)

        return fig
