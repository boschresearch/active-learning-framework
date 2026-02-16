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
from alef.oracles.base_oracle import BaseOracle
from alef.active_learner.base_active_learner import BaseActiveLearner


class BaseOracleActiveLearner(BaseActiveLearner):
    def __init__(self):
        super().__init__()
        self.oracle: BaseOracle = None

    def set_oracle(self, oracle: BaseOracle):
        """
        sets the oracle that should be learned and from which queries should be taken
        """
        self.oracle = oracle

    def sample_ground_truth(self):
        """
        samples many noise-free points from the oracle (only used for plotting)
        """
        self.gt_X, self.gt_function_values = self.oracle.get_random_data(2000, noisy=False)
        self.ground_truth_available = True

    def sample_test_set(self, n_data: int, seed: int = 100, set_seed: bool = False):
        """
        Method for sampling the test dataset directly from the oracle object

        Arguments:
        n_data - int - number of test datapoints that should be sampled
        seed - int - seed for sampling if one should use one
        set_seed - bool - flag if seed should be set
        """
        if set_seed:
            np.random.seed(seed)
        x, y = self.oracle.get_random_data(n_data, noisy=True)
        self.set_test_set(x, y)

    def sample_train_set(self, n_data, seed=100, set_seed=False):
        """
        Method for sampling the initial dataset directly from the oracle object

        Arguments:
        n_data - int - number of initial datapoints that should be sampled
        seed - int - seed for sampling if one should use one
        set_seed - bool - flag if seed should be set
        """
        if set_seed:
            np.random.seed(seed)
        x, y = self.oracle.get_random_data(n_data, noisy=True)
        self.set_train_set(x, y)


if __name__ == "__main__":
    pass
