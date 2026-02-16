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
from typing import Union, Sequence
from copy import deepcopy

from alef.enums.data_structure_enums import OutputType
from alef.utils.utils import create_grid_multi_bounds, check1Dlist, filter_nan
from alef.models.base_model import BaseModel
from alef.oracles.base_oracle import BaseOracle
from alef.oracles.context_interface import ContextSupport


class GPOracleFromData(BaseOracle, ContextSupport):
    def __init__(
        self,
        gp_model: BaseModel,
        x_data: np.ndarray,
        y_data: np.ndarray,
        output_type: OutputType = OutputType.SINGLE_OUTPUT,
    ):
        self._model = gp_model
        self.__x, self.__y = filter_nan(x_data, y_data)
        self.output_type = output_type
        if self.output_type == OutputType.SINGLE_OUTPUT:
            self.__dimension = self.__x.shape[1]
        elif self.output_type == OutputType.MULTI_OUTPUT_FLATTENED:
            self.__dimension = self.__x.shape[1] - 1
            self.__current_p = None
        else:
            raise NotImplementedError
        self._model.infer(self.__x, self.__y)
        self.__a = self.__x[:, : self.get_dimension()].min(axis=0)
        self.__b = self.__x[:, : self.get_dimension()].max(axis=0)

    def set_output_idx(self, p: int):
        if self.output_type == OutputType.MULTI_OUTPUT_FLATTENED:
            self.__current_p = p

    def query(self, x, noisy: bool = True):
        if np.shape(np.atleast_2d(x))[0] > 1:
            raise ValueError("please query only 1 point")
        # if noisy:
        #    print('currently, noisy is a useless argument', file=sys.stderr)
        if self.output_type == OutputType.MULTI_OUTPUT_FLATTENED:
            xx = np.hstack((np.atleast_2d(x), np.array([[self.__current_p]])))
        elif self.output_type == OutputType.SINGLE_OUTPUT:
            xx = np.atleast_2d(x)
        """
        idx = row_wise_compare(self.__x, xx)
        idx = np.where(idx)[0] # np.where return a tuple, idx here is an array

        if len(idx) < 1:
            mu, std = self._model.predictive_dist(xx)
            if noisy:
                return np.random.normal(mu, std)
            else:
                return mu
        elif len(idx) > 1:
            print('Querying observed point, \'noisy\' does not matter', file=sys.stderr)
            idx = np.random.choice(idx)
            return self.__y[idx, 0]
        else:
            print('Querying observed point, \'noisy\' does not matter', file=sys.stderr)
            idx = idx[0]
            return self.__y[idx, 0]
        """
        mu, std = self._model.predictive_dist(xx)
        if noisy:
            return np.random.normal(mu, std)
        else:
            return mu

    def query_multiple_points(self, x, noisy=True):
        if self.output_type == OutputType.MULTI_OUTPUT_FLATTENED:
            x_decorate = np.hstack((x, self.__current_p * np.ones([x.shape[0], 1])))
        elif self.output_type == OutputType.SINGLE_OUTPUT:
            x_decorate = x
        """
        mask = row_wise_compare(x_decorate, self.__x)
        idx = np.where(mask)[0] # np.where return a tuple, idx here is an array

        output = np.empty([x.shape[0], 1])
        for i in idx:
            output[i,0] = self.query(x[None,i,:], noisy=noisy)

        mu, std = self._model.predictive_dist(x_decorate[~mask])
        if noisy:
            output[~mask,0] = np.random.normal(mu, std)
        else:
            output[~mask,0] = mu
        return output
        """
        mu, std = self._model.predictive_dist(x_decorate)
        if noisy:
            return np.random.normal(mu, std)[:, None]
        else:
            return mu[:, None]

    def query_multiple_points_in_sequence(self, x, noisy=True):
        nn = x.shape[0]
        n_infer_max = 2000
        n_round = np.ceil(nn / n_infer_max).astype(int)

        output = np.empty([nn, 1])
        for j in range(n_round):
            output[j * n_infer_max : max(nn + 1, (j + 1) * n_infer_max)] = self.query_multiple_points(
                x[j * n_infer_max : max(nn + 1, (j + 1) * n_infer_max)], noisy=noisy
            )

        return output

    def get_grid_data(self, n_per_dim: int, noisy: bool = True):
        a, b = self.get_box_bounds()
        n = np.array([n_per_dim] * self.get_dimension())
        use_context, context_idx, context_values = self.get_context_status(return_idx=True, return_values=True)
        if use_context:
            a[context_idx] = context_values
            b[context_idx] = context_values
            n[context_idx] = 1
        X = create_grid_multi_bounds(a, b, n)
        # function_values = [np.reshape(self.query(x, noisy), [1,-1]) for x in X]
        # return X, np.concatenate(function_values, axis=0)
        return X, self.query_multiple_points_in_sequence(X, noisy=noisy)

    def get_random_data(self, n, noisy=True):
        a, b = self.get_contextual_box_bound()
        X = np.random.uniform(low=a, high=b, size=(n, self.get_dimension()))
        # function_values = [np.reshape(self.query(x, noisy), [1,-1]) for x in X]
        # return X, np.concatenate(function_values, axis=0)
        return X, self.query_multiple_points_in_sequence(X, noisy=noisy)

    def _get_random_x_in_box(self, n: int, a: Union[float, Sequence[float]], box_width: Union[float, Sequence[float]]):
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

        aa[aa < self.__a] = self.__a[aa < self.__a]
        bb[bb > self.__b] = self.__b[bb > self.__b]

        X = np.random.uniform(low=aa, high=bb, size=(n, self.get_dimension()))
        return X

    def get_random_data_in_box(
        self, n: int, a: Union[float, Sequence[float]], box_width: Union[float, Sequence[float]], noisy: bool = True
    ):
        X = self._get_random_x_in_box(n, a, box_width)

        # function_values = [np.reshape(self.query(x, noisy), [1,-1]) for x in X]
        # return X, np.concatenate(function_values, axis=0)
        return X, self.query_multiple_points_in_sequence(X, noisy=noisy)

    def get_variable_box_bounds(self):
        a, b = self.get_box_bounds()
        use_context, context_idx = self.get_context_status(return_idx=True, return_values=False)
        if use_context:
            return np.delete(a, context_idx), np.delete(b, context_idx)
        else:
            return a, b

    def get_box_bounds(self):
        return deepcopy(self.__a), deepcopy(self.__b)

    def get_contextual_box_bound(self):  # this method is used in alef.pools.engine_pool
        a, b = self.get_box_bounds()
        use_context, context_idx, context_values = self.get_context_status(return_idx=True, return_values=True)
        if use_context:
            a[context_idx] = context_values
            b[context_idx] = context_values
        return a, b

    def get_dimension(self):
        return self.__dimension


if __name__ == "__main__":
    from alef.data_sets.pytest_set import PytestSet
    from alef.configs.kernels.rbf_configs import BasicRBFConfig
    from alef.configs.models.gp_model_config import GPModelFastConfig
    from alef.models.model_factory import ModelFactory

    dataset = PytestSet()
    dataset.load_data_set()
    x, y = dataset.get_complete_dataset()
    model = ModelFactory.build(GPModelFastConfig(kernel_config=BasicRBFConfig(input_dimension=3), optimize_hps=False))
    oracle = GPOracleFromData(model, x, y[:, 0, None])
    print(oracle.get_dimension(), oracle.get_box_bounds())
    print(oracle.get_variable_dimension(), oracle.get_variable_box_bounds())
    oracle.set_context([1], [0.0])
    print(oracle.get_dimension(), oracle.get_box_bounds())
    print(oracle.get_variable_dimension(), oracle.get_variable_box_bounds())
    X, Y = oracle.get_random_data(3000, noisy=False)
    # for i in range(3000):
    #    if not np.allclose( Y[i,0], oracle.query(X[i]) ):
    #        print( f'False at iter {i}, Y={Y[i,0]}, oracle query={oracle.query(X[i])}' )
