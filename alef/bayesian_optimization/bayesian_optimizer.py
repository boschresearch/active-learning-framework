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

from typing import Union
import numpy as np
from alef.acquisition_function.al_acquisition_functions.acq_random import Random
from alef.acquisition_function.bo_acquisition_functions.base_bo_acquisition_function import BaseBOAcquisitionFunction
from alef.utils.metric_curve_plotter import MetricCurvePlotter
from alef.utils.plot_utils import (
    active_learning_1d_plot_with_acquisition,
    active_learning_2d_plot,
    active_learning_2d_plot_without_gt,
    plot_model_specifics,
)
from alef.enums.bayesian_optimization_enums import AcquisitionOptimizationType, ValidationType
from alef.oracles.base_oracle import BaseOracle
from alef.models.base_model import BaseModel
from alef.bayesian_optimization.evolutionary_optimizer import EvolutionaryOptimizer


class BayesianOptimizer:
    """
    Main class for bayesian optimization

    Main Attributes:
        acquisition_function : BaseBOAcquisitionFunction
        validation_type : ValidationType - Enum which validation metric should be used e.g. Simple Regret, Cumm. Regret,...
        model : BaseModel - Model object that is an instance of a child class of BaseModel such as e.g. GPModel
        acquisiton_optimization_type : AcquisitionOptimizationType - specifies how the acquisition function should be optimized
        oracle: BaseOracle - oracle object that should be optimized
    """

    def __init__(
        self,
        acquisition_function: Union[BaseBOAcquisitionFunction, Random],
        validation_type: ValidationType,
        acquisiton_optimization_type: AcquisitionOptimizationType,
        do_plotting: bool,
        random_shooting_n: int,
        steps_evoluationary: int,
        **kwargs,
    ):
        self.validation_type = validation_type
        self.acquisition_function = acquisition_function
        self.acquisiton_optimization_type = acquisiton_optimization_type
        self.validation_metrics = []
        self.ground_truth_available = False
        self.do_plotting = do_plotting
        self.oracle = None
        self.random_shooting_n = random_shooting_n
        self.steps_evoluationary = steps_evoluationary
        self.save_plots = False
        self.plot_path = None
        assert isinstance(self.acquisition_function, BaseBOAcquisitionFunction) or isinstance(
            self.acquisition_function, Random
        )

    def set_oracle(self, oracle: BaseOracle):
        """
        sets oracle that should be optimized

        Arguments:
        oracle - BaseOracle - oracle object that can be queried and that should be optimized
        """
        self.oracle = oracle

    def set_do_plotting(self, do_plotting: bool):
        """
        Method for specifying if plotting should be done

        Arguments:
            do_plotting - bool : flag if plotting should be done
        """
        self.do_plotting = do_plotting

    def set_model(self, model: BaseModel):
        """
        sets surrogate model

        Arguments:
        model - BaseModel - instance of some BaseModel child
        """
        self.model = model

    def sample_ground_truth(self):
        """
        method for sampling many points form the oracle - only for plotting and trivial oracles (with low query times)
        """
        self.ground_truth_available = True
        self.gt_X, self.gt_function_values = self.oracle.get_random_data(2000, noisy=False)

    def sample_from_oracle_to_find_max_value(self, n_data: int, seed: int = 100, set_seed: bool = False):
        """
        method for sampling from the oracle in order to find the true maximum value of the oracle for validation - only for trivial oracles (with low query times)
        """
        if set_seed:
            np.random.seed(seed)
        _, f = self.oracle.get_random_data(n_data, noisy=False)
        self.max_f = np.max(f)

    def save_plots_to_path(self, plot_path: str):
        """
        method to specify that plots are not shown but saved to a path
        """
        self.save_plots = True
        self.plot_path = plot_path

    def set_max_value_for_validation(self, value: float):
        """
        method for manually setting the max value of f - for validation

        Arguments:
        value - float : max value of the oracle (for validation of BO only)
        """
        self.max_f = value

    def sample_train_set(self, n_data: int, seed: int = 100, set_seed: bool = False):
        """
        Method for sampling the initial dataset directly from the oracle object

        Arguments:
        n_data - int - number of initial datapoints that should be sampled
        seed - int - seed for sampling if one should use one
        set_seed - bool - flag if seed should be set
        """
        if set_seed:
            np.random.seed(seed)
        self.x_data, self.y_data = self.oracle.get_random_data(n_data, noisy=True)

    def set_train_set(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Method for setting the train set manually
        Arguments:
        x_train - np.array - array of shape [n,d] containing the inputs of n training datapoints with dimension d
        y_train - np.array - array of shape [n,1] containing the outputs of n training datapoints
        """
        self.x_data = x_train
        self.y_data = y_train

    def update(self):
        """
        Main update function - infers the model on the current dataset, optimizes the acquisition function and returns the query location -
        Acquisition function optimization is specified in self.acquisition_function_optimization_type
        """
        self.model.reset_model()
        self.model.infer(self.x_data, self.y_data)
        box_a, box_b = self.oracle.get_box_bounds()
        dimensions = self.oracle.get_dimension()

        if isinstance(self.acquisition_function, Random):
            query = np.random.uniform(low=box_a, high=box_b, size=(1, dimensions))
            return np.squeeze(query)

        if self.acquisiton_optimization_type == AcquisitionOptimizationType.RANDOM_SHOOTING:
            x_grid = np.random.uniform(low=box_a, high=box_b, size=(self.random_shooting_n, dimensions))
            scores = self.acquisition_func(x_grid)
            new_query = x_grid[np.argmax(scores)]
            return new_query

        if self.acquisiton_optimization_type == AcquisitionOptimizationType.EVOLUTIONARY:
            optimizer = EvolutionaryOptimizer(int(self.random_shooting_n / self.steps_evoluationary))
            new_query, _ = optimizer.maximize(
                self.acquisition_func,
                dimensions,
                np.repeat(box_a, dimensions),
                np.repeat(box_b, dimensions),
                self.steps_evoluationary,
            )
            return new_query

        return None

    def acquisition_func(self, x_grid: np.array) -> np.array:
        """
        Wrapper around acquisition_function object that can be passed to an optimizer like EvolutionaryOptimizer
        """
        scores = self.acquisition_function.acquisition_score(x_grid, self.model, self.x_data, self.y_data)
        return scores

    def maximize(self, n_steps: int):
        """
        Main maximization loop - makes n_steps queries to oracle and returns collected validation metrics and query locations

        Arguments:
        n_steps : int - number of BO steps

        Returns:
            np.array - validation metrics values over the iterations
            np.array - selected queries over the iterations
        """
        self.n_steps = n_steps
        self.validate()
        for i in range(0, self.n_steps):
            query = self.update()
            print("Query")
            print(query)
            new_y = self.oracle.query(query)
            if self.do_plotting:
                self.plot(query, new_y, i)
                # self.plot_validation_curve()
            self.x_data = np.vstack((self.x_data, query))
            self.y_data = np.vstack((self.y_data, [new_y]))
            self.validate()
        return np.array(self.validation_metrics), self.x_data

    def validate(self):
        """
        validation method - calcuates validation metric (self.validation_type specifies which one) and stores it to self.validation_metrics list

        Arguments:
            query : np.array - selected query
        """
        if not self.validation_type == ValidationType.MAX_OBSERVED:
            self.model.reset_model()
            self.model.infer(self.x_data, self.y_data)

        if self.validation_type == ValidationType.SIMPLE_REGRET:
            pred_mus, _ = self.model.predictive_dist(self.x_data)
            simple_regret = self.max_f - np.max(pred_mus)
            self.validation_metrics.append(simple_regret)
        elif self.validation_type == ValidationType.CUMMULATIVE_REGRET:
            pred_mus, _ = self.model.predictive_dist(self.x_data)
            cummulative_regret = self.max_f - np.mean(pred_mus)
            self.validation_metrics.append(cummulative_regret)
        elif self.validation_type == ValidationType.MAX_OBSERVED:
            max_observed = np.max(self.y_data)
            self.validation_metrics.append(max_observed)

    def plot(self, query, new_y, step):
        dimension = self.x_data.shape[1]
        if self.ground_truth_available:
            x_grid = self.gt_X
            y_over_grid = self.gt_function_values
        else:
            box_a, box_b = self.oracle.get_box_bounds()
            x_grid = np.random.uniform(low=box_a, high=box_b, size=(500, dimension))
        plot_name = "query_" + str(step) + ".png"

        if dimension == 1:
            pred_mu, pred_sigma = self.model.predictive_dist(x_grid)
            acquisition_on_grid = self.acquisition_func(x_grid)
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
                    True,
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
        elif dimension == 2:
            pred_mu, pred_sigma = self.model.predictive_dist(x_grid)
            acquisition_on_grid = self.acquisition_func(x_grid)
            if self.ground_truth_available:
                active_learning_2d_plot(
                    x_grid,
                    acquisition_on_grid,
                    pred_mu,
                    y_over_grid,
                    self.x_data,
                    query,
                    save_plot=self.save_plots,
                    file_name=plot_name,
                    file_path=self.plot_path,
                )
            else:
                active_learning_2d_plot_without_gt(
                    x_grid,
                    acquisition_on_grid,
                    pred_mu,
                    self.x_data,
                    query,
                    save_plot=self.save_plots,
                    file_name=plot_name,
                    file_path=self.plot_path,
                )
        else:
            print("Dimension to high for plotting")
        plot_name = "model_specific" + str(step) + ".png"
        plot_model_specifics(
            x_grid, self.x_data, self.model, save_plot=self.save_plots, file_name=plot_name, file_path=self.plot_path
        )

    def plot_validation_curve(self):
        metric_curve_plotter = MetricCurvePlotter(1)
        metric_curve_plotter.add_metrics_curve(
            np.arange(0, len(self.validation_metrics)), self.validation_metrics, "blue", "x", 0, False
        )
        metric_curve_plotter.show()


if __name__ == "__main__":
    pass
