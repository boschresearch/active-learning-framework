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

import time
import numpy as np
from typing import List, Optional
from alef.acquisition_function.al_acquisition_functions.base_al_acquisition_function import BaseALAcquisitionFunction
from alef.utils.plot_utils import (
    active_learning_1d_plot_with_acquisition,
    active_learning_2d_plot,
    active_learning_nd_plot,
    plot_model_specifics,
    active_learning_1d_plot_multioutput,
)
from typing import Tuple
from alef.enums.active_learner_enums import ValidationType
from alef.utils.utils import calculate_multioutput_rmse
from alef.active_learner.base_pool_active_learner import BasePoolActiveLearner


class ActiveLearner(BasePoolActiveLearner):
    """
    Main class for non-batch pool-based active learning - one query at a time - inherits from BasePoolActiveLearner

    Attributes:
        acquisition_function : BaseALAcquisitionFunction
        validation_type : ValidationType - Enum which validation metric should be used e.g. RMSE, NegLoglikeli
        model : BaseModel - Model object that is an instance of a child class of BaseModel such as e.g. GPModel
        use_smaller_acquisition_set : bool - Bool if only a sampled subset of the pool should used for query selection (saves computational budget)
        smaller_set_size : int - Number of samples from the pool used for acquisition calculation
    """

    def __init__(
        self,
        acquisition_function: BaseALAcquisitionFunction,
        validation_type: ValidationType,
        use_smaller_acquistion_set: bool = False,
        smaller_set_size: int = 200,
        validation_at: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()
        self.validation_type = validation_type
        self.validation_at = validation_at
        self.use_smaller_acquistion_set = use_smaller_acquistion_set
        self.smaller_set_size = smaller_set_size
        self.acquisition_function = acquisition_function
        self.inference_time_list = []
        self.acquisition_func_opt_time_list = []
        self.iteration_time_list = []
        self.iteration_time_without_validation_list = []
        assert isinstance(self.acquisition_function, BaseALAcquisitionFunction)

    def reduce_grid(self, grid: np.ndarray, new_grid_size: int):
        """
        Gets a grid of points and reduces the grid - is used to reduce the pool for acquisition function calculation
        """
        print("-Reduce acquisition set ")
        grid_size = grid.shape[0]
        if grid_size > new_grid_size:
            indexes = np.random.choice(list(range(0, grid_size)), new_grid_size, replace=False)
            new_grid = grid[indexes]
            return new_grid
        else:
            return grid

    def update(self):
        """
        Main update function - infers the model on the current dataset, calculates the acquisition function and returns the query location
        """
        time_before_inference = time.perf_counter()
        self.model.reset_model()
        self.model.infer(self.x_data, self.y_data)
        time_after_inference = time.perf_counter()
        x_grid = self.pool.possible_queries()

        if self.use_smaller_acquistion_set:
            x_grid = self.reduce_grid(x_grid, self.smaller_set_size)

        acquisition_function_scores = self.acquisition_func(x_grid)
        new_query = x_grid[np.argmax(acquisition_function_scores)]
        time_after_acquisition_opt = time.perf_counter()
        inference_time = time_after_inference - time_before_inference
        acquisition_opt_time = time_after_acquisition_opt - time_after_inference
        self.inference_time_list.append(inference_time)
        self.acquisition_func_opt_time_list.append(acquisition_opt_time)
        return new_query

    def acquisition_func(self, x_grid: np.array) -> np.array:
        """
        Wrapper around acquisition_function object
        """
        scores = self.acquisition_function.acquisition_score(x_grid, self.model)
        return scores

    def learn(self, n_steps: int, start_index=0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main Active learning loop - makes n_steps queries and updates the model after each query with the new x_data, y_data
        - validation is done after each query

        Arguments:
            n_steps : int - number of active learning iteration/number of queries
            start_index : int - important for plotting and logging - indicates that already start_index-1 AL steps where done previously
        Returns:
            np.array - validation metrics values over the iterations
            np.array - selected queries over the iterations
        """
        self.n_steps = n_steps
        if self.validation_type == ValidationType.RMSE_MULTIOUTPUT:
            self.initialize_multioutput_validation_metrics(self.y_data.shape[1])
        else:
            self.initialize_validation_metrics()
        for i in range(start_index, start_index + self.n_steps):
            time_before_iteration = time.perf_counter()
            query = self.update()
            print("Query")
            print(query)
            new_y = self.pool.query(query)
            time_after_query = time.perf_counter()
            if self.do_plotting:
                self.plot(query, new_y, i)
            self.add_train_point(i, query, new_y)
            if self.validation_at is None or len(self.validation_at) == 0 or i in self.validation_at:
                self.validate(i)
            time_after_iteration = time.perf_counter()
            iteration_time = time_after_iteration - time_before_iteration
            iteration_time_without_validation = time_after_query - time_before_iteration
            self.iteration_time_list.append(iteration_time)
            self.iteration_time_without_validation_list.append(iteration_time_without_validation)
        if self.save_result:
            self.save_experiment_summary()
        return self.validation_metrics, self.x_data

    def get_time_lists(self):
        """
        returns the time lists that were measured

        Returns:
            list - containes inference times over iterations
            list - containes acquisiton opt times over iterations
            list - contains complete iteration time over iterations
            list - containes iteration time without validation - only updata + query time
        """
        return (
            self.inference_time_list,
            self.acquisition_func_opt_time_list,
            self.iteration_time_list,
            self.iteration_time_without_validation_list,
        )

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
        elif self.validation_type == ValidationType.RMSE_MULTIOUTPUT:
            pred_mu, pred_sigma = self.model.predictive_dist(self.x_test)
            rmse_array = calculate_multioutput_rmse(pred_mu, self.y_test)
            self.add_multioutput_validation_array(idx, rmse_array)

    def plot(self, query: np.ndarray, new_y: np.ndarray, step: int):
        """
        Plotting function - gets the actual query and the AL step index und produces plots depending on the input and output dimension
        if self.plots is True the plots are saved to self.plot_path (both variables are set in the parent class)
        """
        dimension = self.x_data.shape[1]
        output_dimension = self.y_data.shape[1]
        x_grid = np.vstack((self.pool.possible_queries(), self.x_data))
        y_over_grid = np.vstack((self.pool.get_y_data(), self.y_data))
        if output_dimension == 1:
            if dimension == 1:
                self.plot_1d(query, new_y, step, x_grid)

            elif dimension == 2:
                pred_mu, pred_sigma = self.model.predictive_dist(x_grid)
                if self.save_plots:
                    plot_name = "query_" + str(step) + ".png"
                    active_learning_2d_plot(
                        x_grid,
                        pred_sigma,
                        pred_mu,
                        y_over_grid,
                        self.x_data,
                        query,
                        save_plot=self.save_plots,
                        file_name=plot_name,
                        file_path=self.plot_path,
                    )
                else:
                    active_learning_2d_plot(x_grid, pred_sigma, pred_mu, y_over_grid, self.x_data, query)
            else:
                if self.save_plots:
                    plot_name = "query_" + str(step) + ".png"
                    active_learning_nd_plot(self.x_data, self.y_data, self.save_plots, plot_name, self.plot_path)
                else:
                    active_learning_nd_plot(self.x_data, self.y_data)

            if self.save_plots:
                plot_name = "model_specific" + str(step) + ".png"
                plot_model_specifics(
                    x_grid,
                    self.x_data,
                    self.model,
                    save_plot=self.save_plots,
                    file_name=plot_name,
                    file_path=self.plot_path,
                )
            else:
                plot_model_specifics(x_grid, self.x_data, self.model)
        else:
            if dimension == 1:
                pred_mu, pred_sigma = self.model.predictive_dist(x_grid)
                active_learning_1d_plot_multioutput(
                    x_grid,
                    pred_mu,
                    pred_sigma,
                    self.x_data,
                    self.y_data,
                    query,
                    new_y,
                    save_plot=False,
                    file_name=None,
                    file_path=None,
                )

    def plot_1d(self, query, new_y, step, x_grid):
        pred_mu, pred_sigma = self.model.predictive_dist(x_grid)
        acquisition_on_grid = self.acquisition_func(x_grid)
        if self.save_plots:
            plot_name = "query_" + str(step) + ".png"
            if self.ground_truth_available:
                active_learning_1d_plot_with_acquisition(
                    x_grid,
                    pred_mu,
                    pred_sigma,
                    acquisition_on_grid,
                    self.x_data,
                    self.y_data,
                    query,
                    new_y,
                    self.ground_truth_available,
                    self.gt_X,
                    self.gt_function_values,
                    save_plot=self.save_plots,
                    file_name=plot_name,
                    file_path=self.plot_path,
                )
            else:
                active_learning_1d_plot_with_acquisition(
                    x_grid,
                    pred_mu,
                    pred_sigma,
                    acquisition_on_grid,
                    self.x_data,
                    self.y_data,
                    query,
                    new_y,
                    save_plot=self.save_plots,
                    file_name=plot_name,
                    file_path=self.plot_path,
                )
        else:
            if self.ground_truth_available:
                active_learning_1d_plot_with_acquisition(
                    x_grid,
                    pred_mu,
                    pred_sigma,
                    acquisition_on_grid,
                    self.x_data,
                    self.y_data,
                    query,
                    new_y,
                    self.ground_truth_available,
                    self.gt_X,
                    self.gt_function_values,
                )
            else:
                active_learning_1d_plot_with_acquisition(
                    x_grid, pred_mu, pred_sigma, acquisition_on_grid, self.x_data, self.y_data, query, new_y
                )
                # pred_mu,pred_cov = self.model.predictive_dist(x_grid)


if __name__ == "__main__":
    pass
