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
from copy import deepcopy
from typing import Sequence

"""
Use this together with BaseOracle (see ./base_oracle.py)

e.g.

class newOracle(BaseOracle, ContextSupport):
    ...
"""


class ContextSupport:
    __use_context = False
    __context_idx = []
    __context_values = []

    def set_context(self, context_idx: Sequence[int] = [], context_values: Sequence[float] = []):
        r"""
        Context are inputs that cannot be manipulated.
        E.g. x is [N, D+2] array, but the last 2 column should be fixed to 1.3 & -0.5, respectively,
            then context_idx = [D, D+1]
                 context_values = [1.3, -0.5]
        This setting is used for contextual Bayesian optimization & contextual active learning.
        Make sure that the context_values are within the box bound of corresponding dim,
            otherwise self.get_random_data_in_box(  ) will report error when checking boundaries
        """
        if len(context_idx) > 0:
            assert len(context_idx) == len(context_values)
            assert max(context_idx) < self.get_dimension()
            assert min(context_idx) >= 0

            self.__use_context = True
            self.__context_idx = context_idx
            self.__context_values = context_values

    def get_context_status(self, return_idx: bool = False, return_values: bool = False):
        r"""
        return current
            context_state (bool), context_idx, context_values
        """
        if return_idx and return_values:
            return self.__use_context, deepcopy(self.__context_idx), deepcopy(self.__context_values)
        elif return_idx:
            return self.__use_context, deepcopy(self.__context_idx)
        elif return_values:
            return self.__use_context, deepcopy(self.__context_values)
        else:
            return self.__use_context

    def _decorate_variable_with_context(self, x_var):
        r"""
        x_var is [N, D-C] array, where C is the context dim

        return
            X [N, D], where context_values are inserted to the correct column
        """
        if self.__use_context:
            xx = np.empty([x_var.shape[0], self.get_dimension()])

            idx = np.ones(self.get_dimension(), dtype=bool)
            idx[self.__context_idx] = False

            xx[..., idx] = x_var
            for i, c_idx in enumerate(self.__context_idx):
                xx[..., c_idx] = self.__context_values[i]

            return xx
        else:
            return x_var

    def get_variable_dimension(self):
        """
        Returns variable dimension of input in the oracle
        This may be less than input dimension when we fixed few inputs to some values

        Returns:
            int - input dimension
        """
        if self.__use_context:
            return self.get_dimension() - len(self.__context_idx)
        else:
            return self.get_dimension()
