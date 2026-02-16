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

from typing import List, Tuple
import numpy as np


class BayesianEnsembleInterface:
    """
    Interface for BaseModels that are hierarchical models and contain posterior samples from the upper hierarchy level like for example
    samples from the hyperparameter posterior for a fully bayesian GP. Than the predictive distribution is a mixture of predictive distributions
    of the lower hierarchies, e.g. a mixture of Gaussians for the fully bayesian GP - the interface specifies methods to access the complete set of
    predictive distribution like for example used for the integrated acquisition function in BO
    """

    def get_predictive_distributions(self, x_test: np.array) -> List[Tuple[np.array, np.array]]:
        """
        Method to get the predictive distributions mean and variance for all hyperparameter posterior samples (evaluated on the grid points)

        Input:
            np.array  input with shape [n,d]
        Output:
            List[Tuple[np.array,np.array]] list containing mean (n,) and variance (n,) arrays of the respective predictive distributions
        """

        raise NotImplementedError
