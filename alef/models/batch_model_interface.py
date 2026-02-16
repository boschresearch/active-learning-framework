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
from abc import ABC, abstractmethod


class BatchModelInterace(ABC):
    """
    Interface a BaseModel needs to implement if it is used for batch active learning
    """

    @abstractmethod
    def entropy_predictive_dist_full_cov(self, x_test: np.array) -> np.float:
        """
        Method for calculating the entropy of the combined predictive distribution for test sequence - used as acquistion function in batch active learning

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        float - entropy of combined predictive distribution
        """
        raise NotImplementedError
