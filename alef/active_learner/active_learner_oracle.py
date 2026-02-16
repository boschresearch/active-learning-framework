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
from typing import List, Optional
from alef.acquisition_function.al_acquisition_functions.acq_random import Random
from alef.acquisition_function.al_acquisition_functions.base_al_acquisition_function import BaseALAcquisitionFunction
from alef.enums.active_learner_enums import ValidationType, OracleALAcquisitionOptimizationType
from alef.utils.plot_utils import active_learning_1d_plot, active_learning_2d_plot

from .base_oracle_active_learner import BaseOracleActiveLearner


class ActiveLearnerOracle(BaseOracleActiveLearner):
    """
    Main class for non-batch oracle-based active learning - one query at a time - collects queries by calling its oracle object

    Main Attributes:
         acquisition_function : BaseALAcquisitionFunction
        validation_type : ValidationType - Enum which validation metric should be used e.g. RMSE, NegLoglikeli
        model : BaseModel - Model object that is an instance of a child class of BaseModel such as e.g. GPModel
        acquisiton_optimization_type : OracleALAcquisitionOptimizationType - specifies how the acquisition function should be optimized
        oracle: BaseOracle - oracle object for which a surrogate model should be learned and which gets called
    """

    def __init__(
        self,
        acquisition_function: BaseALAcquisitionFunction,
        validation_type: ValidationType,
        acquisiton_optimization_type: OracleALAcquisitionOptimizationType,
        validation_at: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()
        self.validation_type = validation_type
        self.validation_at = validation_at
        self.initialize_validation_metrics()
        self.acquisiton_optimization_type = acquisiton_optimization_type
        self.random_shooting_n = 500
        self.acquisition_function = acquisition_function
        assert isinstance(self.acquisition_function, BaseALAcquisitionFunction)

    def update(self):
        """
        Main update function - infers the model on the current dataset, calculates the acquisition function and returns the query location -
        Acquisition function optimization is specified in self.acquisition_function_optimization_type
        TODO: Add evolutionary acquisition function optimizer from BO also here
        """
        self.model.reset_model()
        self.model.infer(self.x_data, self.y_data)
        box_a, box_b = self.oracle.get_box_bounds()
        dimensions = self.oracle.get_dimension()

        if (
            isinstance(self.acquisition_function, Random)
            or self.acquisiton_optimization_type == OracleALAcquisitionOptimizationType.RANDOM_SHOOTING
        ):
            x_grid = np.random.uniform(low=box_a, high=box_b, size=(self.random_shooting_n, dimensions))
            acquisition_function_scores = self.acquisition_function.acquisition_score(x_grid, self.model)
            new_query = x_grid[np.argmax(acquisition_function_scores)]
            return new_query
        else:
            raise NotImplementedError

        return None

    def learn(self, n_steps: int):
        """
        Main Active learning loop - makes n_steps queries (calls the self.oracle object at each query) and updates the model after each query with the new x_data, y_data
        - validation is done after each query

        Arguments:
            n_steps : int - number of active learning iteration/number of queries
        Returns:
            np.array - validation metrics values over the iterations
            np.array - selected queries over the iterations
        """
        if self.do_plotting:
            self.sample_ground_truth()
        self.n_steps = n_steps
        for i in range(0, self.n_steps):
            query = self.update()
            print("Query")
            print(query)
            new_y = self.oracle.query(query)
            if self.do_plotting:
                self.plot(query, new_y, i)
            self.add_train_point(i, query, new_y)
            if self.validation_at is None or len(self.validation_at) == 0 or i in self.validation_at:
                self.validate(i)
        if self.save_result:
            self.save_experiment_summary()
        return self.validation_metrics, self.x_data

    def validate(self, idx: int):
        """
        Validates the currently infered model on the test set - uses the validation_type to choose which metric should be used
        """
        if self.validation_type == ValidationType.RMSE:
            pred_mu, pred_sigma = self.model.predictive_dist(self.x_test)
            rmse = np.sqrt(np.mean(np.power(pred_mu - np.squeeze(self.y_test), 2.0)))
            self.add_validation_value(idx, rmse)
        elif self.validation_type == ValidationType.NEG_LOG_LIKELI:
            log_likelis = self.model.predictive_log_likelihood(self.x_test, self.y_test)
            neg_log_likeli = np.mean(-1 * log_likelis)
            self.add_validation_value(idx, neg_log_likeli)

    def plot(self, query: np.array, new_y: np.float, step: int):
        dimension = self.x_data.shape[1]
        x_grid = self.gt_X
        y_over_grid = self.gt_function_values
        if dimension == 1:
            pred_mu, pred_sigma = self.model.predictive_dist(x_grid)
            if self.save_plots:
                plot_name = "query_" + str(step) + ".png"
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
                    save_plot=self.save_plots,
                    file_name=plot_name,
                    file_path=self.plot_path,
                )
            else:
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
        elif dimension == 2:
            pred_mu, pred_sigma = self.model.predictive_dist(x_grid)
            acquisition_function = pred_sigma
            if self.save_plots:
                plot_name = "query_" + str(step) + ".png"
                active_learning_2d_plot(
                    x_grid,
                    acquisition_function,
                    pred_mu,
                    y_over_grid,
                    self.x_data,
                    query,
                    save_plot=self.save_plots,
                    file_name=plot_name,
                    file_path=self.plot_path,
                )
            else:
                active_learning_2d_plot(x_grid, acquisition_function, pred_mu, y_over_grid, self.x_data, query)
        else:
            print("Dimension too high for plotting")


if __name__ == "__main__":
    pass
