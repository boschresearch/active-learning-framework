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

from typing import Union, Sequence
import os
import sys
import numpy as np
import pandas as pd
import time
from alef.utils.utils import filter_nan
from scipy.stats import norm
from alef.utils.safety_metrices import SafetyAreaMeasure
from alef.utils.metric_curve_plotter import MetricCurvePlotter
from alef.utils.plot_utils import (
    safe_bayesian_optimization_1d_plot,
    safe_bayesian_optimization_2d_plot,
    safe_bayesian_optimization_2d_plot_without_gt_with_only_safepoints,
    safety_function_2d_plot,
    safety_histogram,
)
from alef.enums.data_structure_enums import OutputType
from alef.enums.active_learner_enums import ValidationType
from alef.models.base_model import BaseModel
from alef.acquisition_function.safe_acquisition_functions.base_safe_acquisition_function import (
    BaseSafeAcquisitionFunction,
)
from alef.pools.base_pool import BasePool
from alef.pools.base_pool_with_safety import BasePoolWithSafety


class SafeActiveLearner:
    """
    Main class for safe active learning

    Main Attributes:
        acquisition_function : AcquisitionFunctionType - Enum which acquisiton function type should be performed e.g. SAFE_OPT, SAFE_GP_UCB,...
        validation_type : ValidationType - Enum which validation metric should be used e.g. Simple Regret, Cumm. Regret,...
        do_plotting: bool - whether we plot or not
        model_is_safety_model: bool - whether the safety is constrained directly on the main model or not
        save_results: bool - whether we save the plots/result or not
        experiment_path: str - path where we save files
    """

    def __init__(
        self,
        acquisition_function: BaseSafeAcquisitionFunction,
        validation_type: ValidationType,
        do_plotting: bool,
        query_noisy: bool = True,
        model_is_safety_model: bool = False,
        tolerance: Union[float, Sequence[float]] = 0.01,
        save_results: bool = False,
        experiment_path: str = None,
    ):
        """
        Initialize the SafeActiveLearner.

        Args:
            acquisition_function (BaseSafeAcquisitionFunction): The acquisition function to use.
            validation_type (ValidationType): The type of validation to perform.
            do_plotting (bool): Whether to plot the results.
            query_noisy (bool, optional): Whether the queries are noisy. Defaults to True.
            model_is_safety_model (bool, optional): Whether the model is a safety model. Defaults to False.
            tolerance (Union[float, Sequence[float]], optional): The tolerance for convergence. Defaults to 0.01.
            save_results (bool, optional): Whether to save the results. Defaults to False.
            experiment_path (str, optional): The path to save the results. Defaults to None.
        """
        self.acquisition_function = acquisition_function
        self.validation_type = validation_type
        self.observation_number = []
        self.validation_metrics = []
        self.infer_time = []
        self.validate_time = []
        self.kernel_scale = []
        self.kernel_lengthscales = []
        self.measure_safe_area = False
        self.safe_area = SafetyAreaMeasure()
        """SafetyAreaMeasure compute the areas of predictive safe regions and true safe regions,
                where each regions would be clustered."""
        self.ground_truth_available = False
        self.do_plotting = do_plotting
        self.query_noisy = query_noisy
        self.model_is_safety_model = model_is_safety_model
        self.num_of_models = (
            1 + self.acquisition_function.number_of_constraints - int(self.model_is_safety_model)
        )  # the number of f & q_i, q are safety functions
        self.tolerance = tolerance
        self.save_results = save_results
        self.exp_path = experiment_path
        self.__save_model_pars = False

    def set_pool(self, pool: Union[BasePool, BasePoolWithSafety]):
        """
        Set the pool for the active learner.

        Args:
            pool (Union[BasePool, BasePoolWithSafety]): The pool to set.
        """
        self.pool = pool

    def set_model(self, model: BaseModel, safety_models: Union[BaseModel, Sequence[BaseModel]] = None):
        """
        Set the model for the active learner.

        Args:
            model (BaseModel): The model to set.
            safety_models (Union[BaseModel, Sequence[BaseModel]], optional): The safety models to set. Defaults to None.
        """
        self.model = model
        if self.model_is_safety_model:
            self.safety_models = None
            assert self.acquisition_function.number_of_constraints == 1
            print("ATTENTION - model itself is safety model")
        else:
            if isinstance(safety_models, list):
                self.safety_models = safety_models
            elif isinstance(safety_models, BaseModel):
                self.safety_models = [safety_models]
            assert self.acquisition_function.number_of_constraints == len(self.safety_models)

    def set_train_data(self, x_train, y_train, z_train=None):
        """
        Set the training data.

        Args:
            x_train (np.array): The input training data.
            y_train (np.array): The output training data.
            z_train (np.array, optional): The safety training data. Defaults to None.
        """
        self.x_data = np.atleast_2d(x_train)
        self.y_data = np.atleast_2d(y_train)
        if not self.model_is_safety_model:
            self.z_data = np.atleast_2d(z_train)

    def add_train_data(self, x, y, z=None):
        """
        Add training data.

        Args:
            x (np.array): The input data.
            y (np.array): The output data.
            z (np.array, optional): The safety data. Defaults to None.
        """
        self.x_data = np.vstack((self.x_data, np.atleast_2d(x)))
        self.y_data = np.vstack((self.y_data, np.atleast_2d(y)))
        if not z is None:
            self.z_data = np.vstack((self.z_data, np.atleast_2d(z)))

    def set_test_data(self, x_test, y_test, z_test=None):
        """
        Set the test data.

        Args:
            x_test (np.array): The input test data.
            y_test (np.array): The output test data.
            z_test (np.array, optional): The safety test data. Defaults to None.
        """
        self.x_test = np.atleast_2d(x_test)
        self.y_test = np.atleast_2d(y_test)
        if not self.model_is_safety_model:
            self.z_test = np.atleast_2d(z_test)

    def add_test_data(self, x, y, z=None):
        """
        Add test data.

        Args:
            x (np.array): The input data.
            y (np.array): The output data.
            z (np.array, optional): The safety data. Defaults to None.
        """
        self.x_test = np.vstack((self.x_test, np.atleast_2d(x)))
        self.y_test = np.vstack((self.y_test, np.atleast_2d(y)))
        if not z is None:
            self.z_test = np.vstack((self.z_test, np.atleast_2d(z)))

    def _set_grid_data(self, x_grid, y_grid, z_grid=None):
        """
        Set the grid data for safety checks.

        Args:
            x_grid (np.array): The input grid data.
            y_grid (np.array): The output grid data.
            z_grid (np.array, optional): The safety grid data. Defaults to None.
        """
        self.x_grid = np.atleast_2d(x_grid)
        self.y_grid = np.atleast_2d(y_grid)
        if not self.model_is_safety_model:
            self.z_grid = np.atleast_2d(z_grid)

    def initialize_safe_area_measure(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray = None,
        label_grid: str = None,
        sheet_name: str = None,
    ):
        """
        Initialize the safe area measure.

        Args:
            x_grid (np.ndarray): The input grid data.
            y_grid (np.ndarray): The output grid data.
            z_grid (np.ndarray, optional): The safety grid data. Defaults to None.
            label_grid (str, optional): The label grid. Defaults to None.
            sheet_name (str, optional): The sheet name. Defaults to None.
        """
        self._set_grid_data(x_grid, y_grid, z_grid)
        self.measure_safe_area = True

        d = self.pool.get_dimension()
        self.safe_area.set_object_detector(self.pool.get_dimension())

        if not label_grid is None:
            self.safe_area.true_safe_lands_from_file(label_grid, sheet_name)
        else:
            Z = self.y_grid if self.model_is_safety_model else self.z_grid
            safe_bool = self.acquisition_function.compute_safe_data_set(Z).reshape([-1, 1])
            self.safe_area.true_safe_lands(self.x_grid[..., :d], safe_bool.astype(int))

    def update(self, converge_check: bool):
        """
        Update the active learner by selecting a new query point.

        Args:
            converge_check (bool): Whether to check for convergence.

        Returns:
            np.ndarray: The selected query point.
        """
        self._make_infer()

        x_pool = self.pool.possible_queries()
        idx_dim = self._return_variable_idx()

        acq_score, S = self.acquisition_function.acquisition_score(
            x_pool[:, idx_dim],
            model=self.model,
            safety_models=self.safety_models,
            x_data=self.x_data[:, idx_dim],
            y_data=self.y_data,
            return_safe_set=True,
        )

        if not np.any(S):
            raise StopIteration("There are no safe points to evaluate.")
        converge = 0  # np.all( std[S] <= tolerance)
        if converge_check and converge:
            raise StopIteration("Converge.")

        x_safe = x_pool[S]
        new_query = x_safe[np.argmax(acq_score[S])]

        if self.do_plotting:
            mu, std = self._update_posterior(x_pool)
            self.plotting_booklet = {
                "x_pool": x_pool,
                "safety_mask": S.astype(int),
                "acq_score": acq_score,
                "posterior": (mu, std),
            }
        return new_query

    def learn(self, n_steps: int):
        """
        Perform active learning for a given number of steps.

        Args:
            n_steps (int): The number of steps to perform.

        Returns:
            Tuple[np.ndarray, np.ndarray, int]: The validation metrics, the data, and the number of steps performed.
        """
        self.validate(make_infer=True)
        true_steps = 0
        for i in range(0, n_steps):
            try:
                max_iter = 100
                for j in range(max_iter):
                    query = self.update(converge_check=(i >= 4))
                    print("Query")
                    print(query)

                    if self.measure_safe_area:
                        D = self.pool.get_variable_dimension()
                        print(f"safe region index: {self.safe_area.label_points(np.atleast_2d(query)[:, :D])}")

                    if self.model_is_safety_model:
                        new_y = self.pool.query(query, noisy=self.query_noisy)
                        if np.isnan(new_y) and j < max_iter:
                            print("Qeury is nan, repeat")
                            continue
                        self.add_train_data(query, new_y)
                        break
                    else:
                        new_y, new_z = self.pool.query(query, noisy=self.query_noisy)
                        nan_check = np.isnan(np.concatenate([np.reshape(new_y, -1), np.reshape(new_z, -1)]))
                        if np.all(nan_check) and j < max_iter:
                            print("Qeury is nan, repeat")
                            continue
                        elif np.any(nan_check) and j < max_iter:
                            print("Qeurying output has some nan values, be careful")
                        self.add_train_data(query, new_y, new_z)
                        break

                if self.do_plotting:
                    self.plot(query)

                self.validate(make_infer=False)
                true_steps += 1

            except StopIteration as e:
                print(f"Finish early: {e}")
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt

        return np.array(self.validation_metrics), self.x_data, true_steps

    def validate(self, make_infer: bool = False):
        """
        Validate the model.

        Args:
            make_infer (bool, optional): Whether to make inference before validation. Defaults to False.
        """
        idx_dim = self._return_variable_idx()
        if make_infer:
            self._make_infer()

        print("Validate")
        self.observation_number.append(self.x_data.shape[0])

        t_start = time.perf_counter()

        pred_mu, pred_sigma = self.model.predictive_dist(self.x_test[:, idx_dim])
        if self.validation_type == ValidationType.RMSE:
            rmse = np.sqrt(np.mean(np.power(pred_mu - np.squeeze(self.y_test), 2.0)))
            self.validation_metrics.append(rmse)
        elif self.validation_type == ValidationType.NEG_LOG_LIKELI:
            neg_log_likeli = np.mean(-1 * norm.logpdf(np.squeeze(self.y_test), pred_mu, pred_sigma))
            self.validation_metrics.append(neg_log_likeli)

        t_end = time.perf_counter()
        self.validate_time.append(t_end - t_start)

        if self.measure_safe_area:
            print("Measure safety quality")

            if self.model_is_safety_model:
                S = self.acquisition_function.compute_safe_set(self._get_variable_input(self.x_grid), [self.model])
            else:
                S = self.acquisition_function.compute_safe_set(
                    self._get_variable_input(self.x_grid), self.safety_models
                )
            self.safe_area.true_positive_lands(S.astype(int))
            self.safe_area.false_positive_lands(S.astype(int))

    def plot(self, query):
        """
        Plot the results of the active learning process.

        Args:
            query (np.array): The query made by the active learner.
        """
        if self.ground_truth_available:
            raise NotImplementedError("unfinished")
        x_pool = self.plotting_booklet["x_pool"]
        S_mask = self.plotting_booklet["safety_mask"]
        acq_score = self.plotting_booklet["acq_score"]
        mu, std = self.plotting_booklet["posterior"]

        var_dim = self.pool.get_variable_dimension()
        idx_dim = self._return_variable_idx()

        # len(*) - 1 because we validate once before running the experiment, which means len(*) = num_iter + 1
        if var_dim == 1:
            if self.ground_truth_available:
                raise NotImplementedError("unfinished")
            else:
                z_data = self.z_data if not self.model_is_safety_model else self.y_data
                safe_bayesian_optimization_1d_plot(
                    self.pool.output_type,
                    x_pool[..., idx_dim],
                    acq_score,
                    mu,
                    2 * std,
                    self.acquisition_function.safety_thresholds_lower,
                    self.acquisition_function.safety_thresholds_upper,
                    S_mask,
                    self.x_data[..., idx_dim],
                    self.y_data,
                    z_data,
                    query[..., idx_dim],
                    save_plot=self.save_results,
                    file_name="iter_" + str(len(self.observation_number) - 1) + ".png",
                    file_path=self.exp_path,
                )
        elif var_dim == 2:
            if self.ground_truth_available:
                raise NotImplementedError("unfinished")
                safe_bayesian_optimization_2d_plot("use this function maybe")
            else:
                pred_mu, _ = self.model.predictive_dist(x_pool[:, self._return_variable_idx()])
                safe_bayesian_optimization_2d_plot_without_gt_with_only_safepoints(
                    self.pool.output_type,
                    x_pool[..., idx_dim],
                    acq_score[S_mask.astype(bool)],
                    pred_mu,
                    S_mask,
                    self.x_data[..., idx_dim],
                    self.y_data,
                    query[..., idx_dim],
                    save_plot=self.save_results,
                    file_name="iter_" + str(len(self.observation_number) - 1) + ".png",
                    file_path=self.exp_path,
                )
            z_data = self.z_data if not self.model_is_safety_model else self.y_data
            if self.model_is_safety_model:
                safety_function_2d_plot(
                    self.pool.output_type,
                    x_pool[..., idx_dim],
                    mu,
                    2 * std,
                    self.acquisition_function.safety_thresholds_lower,
                    self.acquisition_function.safety_thresholds_upper,
                    self.x_data[:-1, idx_dim],
                    z_data[:-1],
                    save_plot=self.save_results,
                    file_name="iter_" + str(len(self.observation_number) - 1) + "Z.png",
                    file_path=self.exp_path,
                )
            if not self.model_is_safety_model:
                safety_function_2d_plot(
                    self.pool.output_type,
                    x_pool[..., idx_dim],
                    mu[:, 1:],
                    2 * std[:, 1:],
                    self.acquisition_function.safety_thresholds_lower,
                    self.acquisition_function.safety_thresholds_upper,
                    self.x_data[:-1, idx_dim],
                    z_data[:-1],
                    save_plot=self.save_results,
                    file_name="iter_" + str(len(self.observation_number) - 1) + "Z.png",
                    file_path=self.exp_path,
                )
        else:
            print("Dimension too high for plotting", file=sys.stderr)

    def plot_validation_curve(self, x_start_idx: int = None, filename: str = None):
        """
        Plot the validation curve.

        Args:
            x_start_idx (int, optional): The starting index for the x-axis. Defaults to None.
            filename (str, optional): The filename to save the plot. Defaults to None.
        """
        metric_curve_plotter = MetricCurvePlotter(1)
        if x_start_idx is None:
            x = np.arange(0, len(self.validation_metrics))
        else:
            x = np.arange(x_start_idx, len(self.validation_metrics))
        metric_curve_plotter.add_metrics_curve(
            x, self.validation_metrics[-len(x) :], "blue", self.validation_type.name.lower(), 0, False
        )
        if self.validation_type == ValidationType.RMSE:
            ylim = [-0.05, max(0.5, 1.1 * max(self.validation_metrics))]
            metric_curve_plotter.configure_axes(
                0,
                self.validation_type.name.lower(),
                x_label="iteration",
                y_label="RMSE",
                log_scale_y=False,
                add_legend=True,
                y_lim=ylim,
            )
        else:
            metric_curve_plotter.configure_axes(
                0,
                self.validation_type.name.lower(),
                x_label="iteration",
                y_label=None,
                log_scale_y=False,
                add_legend=True,
            )

        if self.save_results:
            if filename is None:
                filename = "metric_target.png"
            metric_curve_plotter.save_fig(self.exp_path, filename)
        else:
            metric_curve_plotter.show()

    def plot_safe_dist(self, filename="safety_histogram"):
        """
        Plot the safety distribution.

        Args:
            filename (str, optional): The filename to save the plot. Defaults to "safety_histogram".
        """
        z = self.y_data if self.model_is_safety_model else self.z_data
        safety_histogram(
            self.pool.output_type,
            self.x_data,
            z,
            self.acquisition_function.safety_thresholds_lower,
            self.acquisition_function.safety_thresholds_upper,
            save_plot=self.save_results,
            file_name=filename,
            file_path=self.exp_path,
        )

    def plot_safe_area(self, filename="safety_area"):
        """
        Plot the safe area.

        Args:
            filename (str, optional): The filename to save the plot. Defaults to "safety_area".
        """
        self.safe_area.export_plot(save_plot=self.save_results, file_name=filename, file_path=self.exp_path)

    def save_experiment_summary(self, filename="SafeAL_result.csv"):
        """
        Save the experiment summary.

        Args:
            filename (str, optional): The filename to save the summary. Defaults to "SafeAL_result.csv".
        """
        columns = ["iter_idx"]
        columns.extend([f"x{i}" for i in range(self.x_data.shape[1])])
        columns.extend(["y", "safe_bool", self.validation_type.name.lower()])
        columns.extend([f"infer_time_m{i}" for i in range(self.num_of_models)])
        columns.append("validate_time")

        N = self.x_data.shape[0]
        iter_idx = np.empty([N, 1]) * np.nan
        metric = np.empty([N, 1]) * np.nan
        infer_time = np.empty([N, self.num_of_models]) * np.nan
        validate_time = np.empty([N, 1]) * np.nan
        for i, n in enumerate(self.observation_number):
            iter_idx[n - 1] = i
            metric[n - 1] = self.validation_metrics[i]
            infer_time[n - 1, :] = self.infer_time[i]
            validate_time[n - 1] = self.validate_time[i]

        safety_observations = self.y_data if self.model_is_safety_model else self.z_data
        safe_bool = self.acquisition_function.compute_safe_data_set(safety_observations).reshape([-1, 1])

        if self.measure_safe_area:
            D = self.pool.get_variable_dimension()
            safe_area_label = np.empty([self.x_data.shape[0], 1])

            if self.pool.output_type == OutputType.SINGLE_OUTPUT:
                safe_area_label[:, 0] = self.safe_area.label_points(self.x_data[:, :D])
            elif self.pool.output_type == OutputType.MULTI_OUTPUT_FLATTENED:
                safe_area_label[self.x_data[:, -1] == self.pool.task_index, 0] = self.safe_area.label_points(
                    self.x_data[self.x_data[:, -1] == self.pool.task_index, :D]
                )

            data = np.hstack(
                (iter_idx, self.x_data, self.y_data, safe_bool, metric, infer_time, validate_time, safe_area_label)
            )
            columns.append("safe_label_data")
        else:
            data = np.hstack((iter_idx, self.x_data, self.y_data, safe_bool, metric, infer_time, validate_time))

        dataframe = pd.DataFrame(data, columns=columns)
        if self.measure_safe_area:
            n = self.safe_area.get_total_iter_num_true_positive()
            safe_df = self.safe_area.export_df(dataframe[dataframe["iter_idx"] >= iter_idx[-n, 0]].index)
            dataframe = pd.concat([dataframe, safe_df], axis=1)

        # save model k0
        if self.__save_model_pars:
            k_scale = pd.DataFrame(
                columns=[f"model{i}" for i in self.kernel_scale[0].keys()], index=iter_idx[~np.isnan(iter_idx[:, 0]), 0]
            )
            for i, k0_dict in enumerate(self.kernel_scale):
                for key, v in k0_dict.items():
                    k_scale.loc[i, f"model{key}"] = v

        # then save models lengthscale
        if self.__save_model_pars:
            k_lengs = pd.DataFrame(
                columns=pd.MultiIndex.from_tuples(
                    self.kernel_lengthscales[0].keys(), names=["model_idx", "kernel_idx"]
                ),
                index=iter_idx[~np.isnan(iter_idx[:, 0]), 0],
            )
            for i, kl_dict in enumerate(self.kernel_lengthscales):
                for key, v in kl_dict.items():
                    k_lengs.loc[i, key] = v

        if self.save_results:
            dataframe.to_csv(os.path.join(self.exp_path, filename))
            if self.__save_model_pars:
                with pd.ExcelWriter(
                    os.path.join(self.exp_path, filename.split(".csv")[0] + "_model_parameters.xlsx"), mode="w"
                ) as writer:
                    k_scale.to_excel(writer, sheet_name="kernel_scale")
                    k_lengs.to_excel(writer, sheet_name="kernel_lgths")
        else:
            if self.__save_model_pars:
                return dataframe, k_scale, k_lengs
            else:
                return dataframe

    def _make_infer(self):
        """
        Make inference with the current model and data.
        """
        self.model.reset_model()
        idx_dim = self._return_variable_idx()
        infer_time = []

        self.model.infer(*filter_nan(self.x_data[:, idx_dim], self.y_data))

        infer_time.append(self.model.get_last_inference_time())
        if not self.model_is_safety_model:
            for i, model in enumerate(self.safety_models):
                model.reset_model()
                model.infer(*filter_nan(self.x_data[:, idx_dim], self.z_data[:, i, None]))
                infer_time.append(model.get_last_inference_time())

        self.infer_time.append(tuple(infer_time))

        if self.__save_model_pars:
            self._track_model_parameters()

    def _track_model_parameters(self):
        """
        Track the model parameters.
        """
        k0 = {0: self.model.model.kernel.prior_scale}
        if not self.model_is_safety_model:
            for j, sm in enumerate(self.safety_models):
                k0[j] = sm.model.kernel.prior_scale

        self.kernel_scale.append(k0)

        models = [self.model]
        if not self.model_is_safety_model:
            models.extend(self.safety_models)

        if self.pool.output_type == OutputType.SINGLE_OUTPUT:
            self.kernel_lengthscales.append(
                {
                    (i, 0): tuple(
                        m.model.kernel.kernel.lengthscales.numpy(),
                    )
                    for i, m in enumerate(models)
                }
            )
        elif self.pool.output_type == OutputType.MULTI_OUTPUT_FLATTENED:
            kl_dict = {}
            for i, m in enumerate(models):
                for j, k in enumerate(m.model.kernel.latent_kernels):
                    kl_dict[(i, j)] = tuple(k.lengthscales.numpy())
            self.kernel_lengthscales.append(kl_dict)

    def _return_variable_idx(self):
        """
        Return the variable indices.

        Returns:
            np.ndarray: The variable indices.
        """
        idx_dim = np.ones(self.x_data.shape[1], dtype=bool)

        dim = self.pool.get_dimension()
        var_dim = self.pool.get_variable_dimension()
        if var_dim < dim:
            _, idx = self.pool.get_context_status(return_idx=True)
            idx_dim[idx] = False

        return idx_dim

    def _get_variable_input(self, x: np.ndarray):
        """
        Get the variable input from the data.

        Args:
            x (np.ndarray): The input data.

        Returns:
            np.ndarray: The variable input.
        """
        idx_dim = self._return_variable_idx()
        return x[:, idx_dim]

    def _update_posterior(self, x: np.ndarray):
        """
        Update the posterior distribution.

        Args:
            x (np.ndarray): The input data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The mean and standard deviation of the posterior distribution.
        """
        mu = np.empty((x.shape[0], self.num_of_models), dtype=np.float)
        std = np.empty((x.shape[0], self.num_of_models), dtype=np.float)

        idx_dim = self._return_variable_idx()

        mu[:, 0], std[:, 0] = self.model.predictive_dist(x[:, idx_dim])
        for i in range(1, self.num_of_models):  # if self.model_is_safety_model, self.num_of_models will be 1 anyways
            mu[:, i], std[:, i] = self.safety_models[i - 1].predictive_dist(x[:, idx_dim])
        return mu, std


if __name__ == "__main__":
    pass
