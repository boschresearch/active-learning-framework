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
from scipy.stats import norm
from alef.utils.plot_utils import active_learning_1d_plot, active_learning_2d_plot
from alef.enums.bayesian_optimization_enums import AcquisitionFunctionType, AcquisitionOptimizationType, ValidationType
from alef.oracles.base_oracle import BaseOracle
from alef.models.base_model import BaseModel
from alef.bayesian_optimization.evolutionary_optimizer import EvolutionaryOptimizer
from typing import List


class HedgeBayesianOptimizerOracle:
    def __init__(
        self,
        acquisition_function_type: AcquisitionFunctionType,
        validation_type: ValidationType,
        acquisiton_optimization_type: AcquisitionOptimizationType,
        do_plotting: bool,
    ):
        self.acquisition_function_type = acquisition_function_type
        self.validation_type = validation_type
        self.acquisiton_optimization_type = acquisiton_optimization_type
        self.validation_metrics = []
        self.ground_truth_available = False
        self.do_plotting = do_plotting
        self.oracle = None
        self.random_shooting_n = 500
        self.steps_evoluationary = 5
        self.gp_ucb_beta = 3.0
        self.ei_xi = 0.01

    def set_oracle(self, oracle: BaseOracle):
        """
        Set the oracle.
        """
        self.oracle = oracle

    def set_models(self, model_list: List[BaseModel]):
        """
        Set the list of models.
        """
        self.model_list = model_list

    def sample_ground_truth(self):
        """
        Sample the ground truth data.
        """
        self.ground_truth_available = True
        self.gt_X, self.gt_function_values = self.oracle.get_random_data(2000, noisy=False)

    def sample_from_oracle_to_find_max_value(self, n_data, seed=100, set_seed=False):
        """
        Sample data from the oracle to find the maximum value.
        """
        if set_seed:
            np.random.seed(seed)
        _, f = self.oracle.get_random_data(n_data, noisy=False)
        self.max_f = np.max(f)

    def set_max_value_for_validation(self, value):
        """
        Set the maximum value for validation.
        """
        self.max_f = value

    def sample_train_set(self, n_data, seed=100, set_seed=False):
        """
        Sample the training set.
        """
        if set_seed:
            np.random.seed(seed)
        self.x_data, self.y_data = self.oracle.get_random_data(n_data, noisy=True)

    def set_train_set(self, x_train, y_train):
        """
        Set the training set.
        """
        self.x_data = x_train
        self.y_data = y_train

    def select_query(self, query_list):
        """
        Select a query from the query list.
        """
        sample_weights_unnormalized = np.exp(self.gs * self.nu)
        weights = sample_weights_unnormalized / np.sum(sample_weights_unnormalized)
        print(weights)
        query_index = np.random.choice(np.arange(len(query_list)), 1, replace=False, p=weights)[0]
        print(query_index)
        return query_list[query_index], query_index

    def update_selection_variables(self, query_list):
        """
        Update the selection variables.
        """
        for i, query in enumerate(query_list):
            reward = self.model_list[i].predictive_dist(np.array([query]))[0]
            self.gs[i] = self.gs[i] + reward

    def initialize_selection_variables(self):
        """
        Initialize the selection variables.
        """
        self.gs = np.zeros(len(self.model_list))
        self.nu = 0.2

    def infer_models(self):
        """
        Infer the models.
        """
        for model in self.model_list:
            model.reset_model()
            model.infer(self.x_data, self.y_data)

    def get_query(self, model: BaseModel, infer_model=False):
        """
        Get a query from the model.
        """
        if infer_model:
            model.reset_model()
            model.infer(self.x_data, self.y_data)
        box_a, box_b = self.oracle.get_box_bounds()
        dimensions = self.oracle.get_dimension()

        if self.acquisition_function_type == AcquisitionFunctionType.RANDOM:
            query = np.random.uniform(low=box_a, high=box_b, size=(1, dimensions))
            return np.squeeze(query)

        if self.acquisiton_optimization_type == AcquisitionOptimizationType.RANDOM_SHOOTING:
            x_grid = np.random.uniform(low=box_a, high=box_b, size=(self.random_shooting_n, dimensions))
            acquisition_function = self.acquisition_function(model)(x_grid)
            new_query = x_grid[np.argmax(acquisition_function)]
            return new_query

        if self.acquisiton_optimization_type == AcquisitionOptimizationType.EVOLUTIONARY:
            optimizer = EvolutionaryOptimizer(int(self.random_shooting_n / self.steps_evoluationary))
            new_query, _ = optimizer.maximize(
                self.acquisition_function(model),
                dimensions,
                np.repeat(box_a, dimensions),
                np.repeat(box_b, dimensions),
                self.steps_evoluationary,
            )
            return new_query

        return None

    def acquisition_function(self, model: BaseModel):
        """
        Define the acquisition function.
        """
        acquisition_function_type = self.acquisition_function_type

        def func(x_grid):
            if acquisition_function_type == AcquisitionFunctionType.GP_UCB:
                pred_mu, pred_sigma = model.predictive_dist(x_grid)
                acquisition_function_value = pred_mu + np.sqrt(self.gp_ucb_beta) * pred_sigma
            elif acquisition_function_type == AcquisitionFunctionType.EXPECTED_IMPROVEMENT:
                pred_mu_data, _ = model.predictive_dist(self.x_data)
                current_max = np.max(pred_mu_data)
                pred_mu_grid, pred_sigma_grid = model.predictive_dist(x_grid)
                d = pred_mu_grid - current_max - self.ei_xi
                acquisition_function_value = d * norm.cdf(d / pred_sigma_grid) + pred_sigma_grid * norm.pdf(
                    d / pred_sigma_grid
                )
            return acquisition_function_value

        return func

    def maximize(self, n_steps):
        """
        Maximize the objective function.
        """
        self.n_steps = n_steps
        self.initialize_selection_variables()
        self.infer_models()
        for i in range(0, self.n_steps):
            query_list = [self.get_query(model) for model in self.model_list]
            query, query_index = self.select_query(query_list)
            print("Query")
            print(query)
            new_y = self.oracle.query(query)
            if self.do_plotting:
                self.plot(query, query_index, new_y)
            self.x_data = np.vstack((self.x_data, query))
            self.y_data = np.vstack((self.y_data, [new_y]))
            self.infer_models()
            self.validate(query)
            self.update_selection_variables(query_list)
        return np.array(self.validation_metrics), self.x_data

    def validate(self, query):
        """
        Validate the model.
        """
        if self.validation_type == ValidationType.SIMPLE_REGRET:
            pred_mu_maxs = [np.max(model.predictive_dist(self.x_data)[0]) for model in self.model_list]
            simple_regret = self.max_f - np.max(pred_mu_maxs)
            self.validation_metrics.append(simple_regret)
        elif self.validation_type == ValidationType.CUMMULATIVE_REGRET:
            pred_mu_means = [np.mean(model.predictive_dist(self.x_data)[0]) for model in self.model_list]
            cummulative_regret = self.max_f - np.max(pred_mu_means)
            self.validation_metrics.append(cummulative_regret)

    def plot(self, query, query_index, new_y):
        """
        Plot the results.
        """
        dimension = self.x_data.shape[1]
        if self.ground_truth_available:
            x_grid = self.gt_X
            y_over_grid = self.gt_function_values
        else:
            box_a, box_b = self.oracle.get_box_bounds()
            x_grid = np.random.uniform(low=box_a, high=box_b, size=(500, dimension))

        if dimension == 1:
            pred_mu, pred_sigma = self.model_list[query_index].predictive_dist(x_grid)
            if self.ground_truth_available:
                active_learning_1d_plot(
                    x_grid,
                    pred_mu,
                    pred_sigma,
                    self.x_data,
                    self.y_data,
                    query,
                    new_y,
                    True,
                    self.gt_X,
                    self.gt_function_values,
                )
            else:
                active_learning_1d_plot(x_grid, pred_mu, pred_sigma, self.x_data, self.y_data, query, new_y)
        elif dimension == 2:
            if self.ground_truth_available:
                pred_mu, pred_sigma = self.model_list[query_index].predictive_dist(x_grid)
                acquisition_function = pred_sigma
                active_learning_2d_plot(x_grid, acquisition_function, pred_mu, y_over_grid, self.x_data, query)
        else:
            print("Dimension to high for plotting")


if __name__ == "__main__":
    pass
