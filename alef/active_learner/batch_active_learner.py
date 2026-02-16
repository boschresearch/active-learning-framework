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
from alef.acquisition_function.al_acquisition_functions.base_batch_al_acquisition_function import (
    BaseBatchALAcquisitionFunction,
)
from alef.utils.plot_utils import (
    active_learning_1d_plot,
    active_learning_2d_plot,
    active_learning_nd_plot,
    plot_model_specifics,
    active_learning_1d_plot_multioutput,
)
from typing import Tuple
from alef.enums.active_learner_enums import (
    ValidationType,
    BatchAcquisitionOptimizationType,
)
from alef.models.base_model import BaseModel
from alef.utils.utils import calculate_multioutput_rmse
from alef.active_learner.base_pool_active_learner import BasePoolActiveLearner
from alef.models.batch_model_interface import BatchModelInterace


class BatchActiveLearner(BasePoolActiveLearner):
    def __init__(
        self,
        acquisition_function: BaseALAcquisitionFunction,
        validation_type: ValidationType,
        batch_size: int,
        use_smaller_acquistion_set: bool = True,
        smaller_set_size: int = 200,
        validation_at: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Initialize the BatchActiveLearner.

        Args:
            acquisition_function (BaseALAcquisitionFunction): The acquisition function to use.
            validation_type (ValidationType): The type of validation to perform.
            batch_size (int): The size of the batch.
            use_smaller_acquistion_set (bool, optional): Whether to use a smaller acquisition set. Defaults to True.
            smaller_set_size (int, optional): The size of the smaller acquisition set. Defaults to 200.
            validation_at (Optional[List[int]], optional): The steps at which to perform validation. Defaults to None.
        """
        super().__init__()
        self.validation_type = validation_type
        self.validation_at = validation_at
        self.batch_size = batch_size
        self.optimization_type = BatchAcquisitionOptimizationType.GREEDY
        self.use_smaller_acquistion_set = use_smaller_acquistion_set
        self.smaller_set_size = smaller_set_size
        self.acquisition_function = acquisition_function
        assert isinstance(self.acquisition_function, BaseBatchALAcquisitionFunction) or isinstance(
            self.acquisition_function, Random
        )

    def set_model(self, model: BaseModel):
        """
        Set the model for the active learner.

        Args:
            model (BaseModel): The model to set.
        """
        assert isinstance(model, BatchModelInterace)
        super().set_model(model)

    def reduce_grid(self, grid: np.ndarray, new_grid_size: int):
        """
        Reduce the grid of points to a smaller size.

        Args:
            grid (np.ndarray): The grid of points.
            new_grid_size (int): The new size of the grid.

        Returns:
            np.ndarray: The reduced grid.
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
        Update the active learner by selecting a new batch of points.

        Returns:
            np.ndarray: The selected batch of points.
        """
        self.model.reset_model()
        self.model.infer(self.x_data, self.y_data)
        x_grid = self.pool.possible_queries()

        if self.use_smaller_acquistion_set:
            x_grid = self.reduce_grid(x_grid, self.smaller_set_size)

        if isinstance(self.acquisition_function, Random):
            indexes = np.random.choice(list(range(0, x_grid.shape[0])), self.batch_size, replace=False)
            return x_grid[indexes]

        def acquisition_func(batch):
            return self.acquisition_function.acquisition_score(batch, self.model)

        batch = self.optimize(acquisition_func, x_grid)
        return batch

    def optimize(self, acquisition_func, x_grid):
        """
        Optimize the acquisition function to select the best batch of points.

        Args:
            acquisition_func (callable): The acquisition function to optimize.
            x_grid (np.ndarray): The grid of points to consider.

        Returns:
            np.ndarray: The selected batch of points.
        """
        x_grid = x_grid.copy()
        if self.optimization_type == BatchAcquisitionOptimizationType.GREEDY:
            batch = []
            for i in range(0, self.batch_size):
                print("Batch:")
                print(batch)
                acquistion_set = []
                counter = 0
                for x in x_grid:
                    possible_batch = np.array(batch + [x])
                    acquisition_func_value = acquisition_func(possible_batch)
                    acquistion_set.append(acquisition_func_value)
                    counter += 1
                    if counter % 50 == 0:
                        print(str(counter + 1) + "/" + str(len(x_grid)) + " grid points calculated")
                assert len(acquistion_set) == counter
                best_index = np.argmax(np.array(acquistion_set))
                selected_query = x_grid[best_index]
                x_grid = np.delete(x_grid, best_index, axis=0)
                print("Selected Query:")
                print(selected_query)
                batch.append(selected_query)
            return np.array(batch)

    def learn(self, n_steps: int, start_index=0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform active learning for a given number of steps.

        Args:
            n_steps (int): The number of steps to perform.
            start_index (int, optional): The starting index for the steps. Defaults to 0.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The validation metrics and the data.
        """
        self.n_steps = n_steps
        if self.validation_type == ValidationType.RMSE_MULTIOUTPUT:
            self.initialize_multioutput_validation_metrics(self.y_data.shape[1])
        else:
            self.initialize_validation_metrics()
        for i in range(start_index, start_index + self.n_steps):
            queries = self.update()
            print("Queries:")
            print(queries)
            y_queries = []
            for query in queries:
                new_y = self.pool.query(query)
                y_queries.append(new_y)
            y_queries = np.array(y_queries)
            if self.do_plotting:
                self.plot(queries, y_queries, i)
            self.add_batch_train_point(i, queries, y_queries)
            print(self.x_data)
            print(self.y_data)
            self.validate(i)
        return self.validation_metrics, self.x_data

    def validate(self, idx: int):
        """
        Validate the model at a given step.

        Args:
            idx (int): The step index.
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

    def plot(self, queries: np.ndarray, new_y: np.ndarray, step: int):
        """
        Plot the results of the active learning process.

        Args:
            queries (np.ndarray): The queries made by the active learner.
            new_y (np.ndarray): The new data points obtained.
            step (int): The current step of the active learning process.
        """
        dimension = self.x_data.shape[1]
        output_dimension = self.y_data.shape[1]
        x_grid = np.vstack((self.pool.possible_queries(), self.x_data))
        y_over_grid = np.vstack((self.pool.get_y_data(), self.y_data))
        if output_dimension == 1:
            if dimension == 1:
                pred_mu, pred_sigma = self.model.predictive_dist(x_grid)
                if self.save_plots:
                    plot_name = "query_" + str(step) + ".png"
                    if self.ground_truth_available:
                        active_learning_1d_plot(
                            x_grid,
                            pred_mu,
                            pred_sigma,
                            self.x_data,
                            self.y_data,
                            queries,
                            new_y,
                            self.ground_truth_available,
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
                            queries,
                            new_y,
                            save_plot=self.save_plots,
                            file_name=plot_name,
                            file_path=self.plot_path,
                        )
                else:
                    if self.ground_truth_available:
                        active_learning_1d_plot(
                            x_grid,
                            pred_mu,
                            pred_sigma,
                            self.x_data,
                            self.y_data,
                            queries,
                            new_y,
                            self.ground_truth_available,
                            self.gt_X,
                            self.gt_function_values,
                        )
                    else:
                        active_learning_1d_plot(x_grid, pred_mu, pred_sigma, self.x_data, self.y_data, queries, new_y)

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
                        queries,
                        save_plot=self.save_plots,
                        file_name=plot_name,
                        file_path=self.plot_path,
                    )
                else:
                    active_learning_2d_plot(x_grid, pred_sigma, pred_mu, y_over_grid, self.x_data, queries)
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
                    queries,
                    new_y,
                    save_plot=False,
                    file_name=None,
                    file_path=None,
                )
                # pred_mu,pred_cov = self.model.predictive_dist(x_grid)


if __name__ == "__main__":
    pass
