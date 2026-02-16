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
from alef.pools.standard_pool import Pool
from alef.active_learner.base_active_learner import BaseActiveLearner


class BasePoolActiveLearner(BaseActiveLearner):
    """
    Base class for all pool active learner - holds a pool oject and provides setter and getter for initial data and test data
    Main functionality such as update of the model, the learning procedure or the valudation needs to be implemented by the child classes

    Attributes:
        pool - Pool : Pool object that is populated by data and that can be queried
        validation_metrics - List : list of values of the validation over the queries
        x_data - np.array : input data - to initialize can be sampled from the pool or set manually
        y_data - np.array : output data
        x_test - np.array : test input data - needs to be set manually
        y_test - np.array : test output data

    """

    def __init__(self):
        super().__init__()
        self.pool: Pool = Pool()
        self.x_data: np.array
        self.y_data: np.array
        self.x_test: np.array
        self.y_test: np.array

    def set_ground_truth(self, gt_X: np.ndarray, gt_function_values: np.ndarray):
        self.ground_truth_available = True
        self.gt_X = gt_X
        self.gt_function_values = gt_function_values

    def set_pool(self, complete_x_data: np.ndarray, complete_y_data: np.ndarray):
        """
        Method for population the pool with data

        Arguments:
            complete_x_data - np.array : complete input data of the pool
            complete_y_data - np.array : complete output data of the pool
        """
        self.pool.set_data(complete_x_data, complete_y_data)

    def get_data_sets(self):
        """
        Method to get the data set currently present in the active learner
        """
        return self.x_data, self.y_data

    def sample_initial_data(self, n_data, seed=100, set_seed=False):
        """
        Method to sample initial data from the pool - can be seeded for reproducible experiments
        """
        x, y = self.pool.sample_random(n_data, seed, set_seed)
        self.set_train_set(x, y)

    def set_initial_queries_manually(self, x_data: np.ndarray):
        """
        Method for only setting the initial x values manually - those are then queried from the pool
        """
        query = x_data[0]
        y_data = np.array([self.pool.query(query)])
        for i in range(1, x_data.shape[0]):
            query = x_data[i]
            new_y = self.pool.query(query)
            y_data = np.append(y_data, [new_y])
        y_data = np.expand_dims(y_data, axis=1)
        self.set_train_set(x_data, y_data)

    def set_initial_dataset_manually(self, x_data: np.ndarray, y_data: np.ndarray):
        """
        Method for setting the initial dataset manually (x and y data)
        """
        self.set_train_set(x_data, y_data)
