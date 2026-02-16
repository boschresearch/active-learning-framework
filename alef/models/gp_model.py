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
import traceback
from matplotlib.colors import NoNorm
import numpy as np
from typing import Tuple, Optional
import gpflow
from gpflow.utilities import print_summary, set_trainable
from tensorflow_probability import distributions as tfd
from alef.kernels.input_initialized_kernel_interface import InputInitializedKernelInterface
from alef.utils.gp_paramater_cache import GPParameterCache
from alef.utils.plotter2D import Plotter2D
from alef.utils.utils import normal_entropy
from alef.models.base_model import BaseModel
from scipy.stats import multivariate_normal
from alef.models.batch_model_interface import BatchModelInterace
from alef.kernels.regularized_kernel_interface import RegularizedKernelInterface
from scipy.stats import norm
from alef.enums.global_model_enums import PredictionQuantity, InitialParameters
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class GPModel(BaseModel, BatchModelInterace):
    """
    Class that implements standard Gaussian process regression with Type-2 ML for kernel hyperparameter infernece. It forms mainly a wrapper around
    the gpflow.models.GPR object

    Attributes:
        kernel: kernel that is used inside the Gaussian process
        model: holds the gpflow.models.GPR instance
        optimize_hps: bool if kernel parameters are trained
        train_likelihood_variance: bool if likelihood variance is trained
        observation_noise: observation noise level - is either set fixed to that value or acts as initial starting value for optimization
        pertube_parameters_at_start: bool if parameters of the kernels should be pertubed before optimization
        initial_parameter_strategy: InitialParameters determines how the initial trainable parameters are sampled before the start of optimization
        perform_multi_start_optimization: bool if multiple initial values should be used for optimization
        set_prior_on_observation_noise: bool if prior should be applied to obvservation noise (Exponential prior with expected value self.observation_noise)
    """

    def __init__(
        self,
        kernel: gpflow.kernels.Kernel,
        observation_noise: float,
        optimize_hps: bool,
        train_likelihood_variance: bool,
        sample_initial_parameters_at_start: bool,
        initial_parameter_strategy: InitialParameters,
        perturbation_for_multistart_opt: float,
        perturbation_for_singlestart_opt: float,
        initial_uniform_lower_bound: float,
        initial_uniform_upper_bound: float,
        perform_multi_start_optimization: bool,
        set_prior_on_observation_noise: bool,
        n_starts_for_multistart_opt: int,
        expected_observation_noise: float,
        prediction_quantity: PredictionQuantity,
        **kwargs,
    ):
        self.kernel = gpflow.utilities.deepcopy(kernel)
        self.kernel_initial_parameter_cache = GPParameterCache()
        if not isinstance(self.kernel, InputInitializedKernelInterface):
            self.kernel_initial_parameter_cache.store_parameters_from_model(self.kernel)
        self.observation_noise = observation_noise
        self.sample_initial_parameters_at_start = sample_initial_parameters_at_start
        self.model = None
        self.optimize_hps = optimize_hps
        self.train_likelihood_variance = train_likelihood_variance
        self.use_mean_function = False
        self.last_opt_time = None
        self.last_multi_start_opt_time = None
        self.initial_parameter_strategy = initial_parameter_strategy
        self.initial_uniform_lower_bound = initial_uniform_lower_bound
        self.initial_uniform_upper_bound = initial_uniform_upper_bound
        self.perform_multi_start_optimization = perform_multi_start_optimization
        self.n_starts_for_multistart_opt = n_starts_for_multistart_opt

        if self.perform_multi_start_optimization:
            self.perturbation_factor = perturbation_for_multistart_opt
        else:
            self.perturbation_factor = perturbation_for_singlestart_opt

        self.set_prior_on_observation_noise = set_prior_on_observation_noise
        self.expected_observation_noise = expected_observation_noise
        self.prediction_quantity = prediction_quantity
        self.print_summaries = True
        if isinstance(kernel, RegularizedKernelInterface):
            self.add_kernel_hp_regularizer = True
        else:
            self.add_kernel_hp_regularizer = False

    def set_kernel(self, kernel):
        """
        Method to manually set the kernel after initialization of the object

        Arguments:
            kernel gpflow.kernels.Kernel object
        """
        self.kernel = gpflow.utilities.deepcopy(kernel)
        self.kernel_initial_parameter_cache = GPParameterCache()
        if not isinstance(self.kernel, InputInitializedKernelInterface):
            self.kernel_initial_parameter_cache.store_parameters_from_model(self.kernel)
        if isinstance(kernel, RegularizedKernelInterface):
            self.add_kernel_hp_regularizer = True
        else:
            self.add_kernel_hp_regularizer = False

    def get_kernel(self, deep_copy: bool = False):
        if deep_copy:
            return gpflow.utilities.deepcopy(self.kernel)
        return self.kernel

    def get_learned_observation_noise(self) -> np.float:
        assert self.model is not None
        return np.sqrt(self.model.likelihood.variance.numpy())

    def set_initial_observation_noise(self, observation_noise):
        self.observation_noise = observation_noise

    def set_optimize_hps(self, optimize_hps: bool):
        self.optimize_hps = optimize_hps

    def fix_current_hyperparameters(self):
        """
        Sets current kernel and observation noise to default initial value
        """
        logger.info("Fix GP hyperparameters")
        self.set_kernel(self.kernel)
        self.set_initial_observation_noise(self.get_learned_observation_noise())
        self.set_optimize_hps(False)
        self.print_model_summary()

    def get_marginal_likelihood_values_over_2d_grid(
        self,
        range_dim1: Tuple[float, float],
        range_dim2: Tuple[float, float],
        range_dim12: Tuple[float, float] = None,
        range_dim22: Tuple[float, float] = None,
        add_second_ranges=False,
        n_grid=300,
        transform_1=lambda x: x,
        transform_2=lambda x: x,
    ):
        assert len(self.model.trainable_parameters) == 2
        param_to_name = {param: name for name, param in gpflow.utilities.parameter_dict(self.model).items()}
        copied_model = gpflow.utilities.deepcopy(self.model)
        assert isinstance(copied_model, gpflow.models.gpr.GPR)
        x_values = []
        f_values = []
        if add_second_ranges:
            n_grid = 2 * n_grid
        for i in range(0, n_grid):
            if add_second_ranges and i > n_grid / 2:
                x1 = np.random.uniform(range_dim12[0], range_dim12[1], size=(1,))
                x2 = np.random.uniform(range_dim22[0], range_dim22[1], size=(1,))
            else:
                x1 = np.random.uniform(range_dim1[0], range_dim1[1], size=(1,))
                x2 = np.random.uniform(range_dim2[0], range_dim2[1], size=(1,))
            shape_1 = copied_model.trainable_parameters[0].shape
            shape_2 = copied_model.trainable_parameters[1].shape
            copied_model.trainable_parameters[0].assign(x1.reshape(shape_1))
            copied_model.trainable_parameters[1].assign(x2.reshape(shape_2))
            f = copied_model.log_marginal_likelihood()
            x_values.append(np.array([transform_1(x1), transform_2(x2)]))
            f_values.append(f)
        x_values = np.squeeze(np.array(x_values))
        f_values = np.array(f_values)
        param_name_1 = param_to_name[self.model.trainable_parameters[0]]
        param_name_2 = param_to_name[self.model.trainable_parameters[1]]
        return x_values, f_values, param_name_1, param_name_2

    def plot_marginal_likelihood_landscape(
        self,
        x_values,
        f_values,
        param_name_1,
        param_name_2,
        return_plotter=False,
        norm=NoNorm(),
        center_around_quantile=False,
        center_quantile=0.5,
    ):
        plotter = Plotter2D(1)
        if center_around_quantile:
            f_values = f_values - np.quantile(f_values, center_quantile)
            f_for_levels = f_values * 10.0
            levels = (
                [np.min(f_for_levels)]
                + [np.quantile(f_for_levels, i) for i in np.linspace(0.0001, center_quantile, 1000)]
                + [np.quantile(f_for_levels, i) for i in np.linspace(center_quantile + 0.0001, 0.9999, 1000)]
                + [np.max(f_for_levels)]
            )
        else:
            levels = 500
        title = "log marginal likelihood"
        # plotter.add_gt_function(x_values, f_values, "viridis", levels, 0)
        contour = plotter.give_axes(0, 0).tricontourf(
            np.squeeze(x_values[:, 0]), np.squeeze(x_values[:, 1]), f_values, levels=levels, cmap="viridis", norm=norm
        )
        plotter.fig.colorbar(contour, ax=plotter.give_axes(0, 0))
        x_label = param_name_1
        y_label = param_name_2
        plotter.configure_axes(0, 0, title, x_label, y_label)
        if return_plotter:
            return plotter
        else:
            plotter.show()

    def reset_model(self):
        """
        resets the model to the initial values - kernel parameters and observation noise are reset to initial values - gpflow model is deleted
        """
        if self.model is not None:
            if not isinstance(self.kernel, InputInitializedKernelInterface):
                self.kernel_initial_parameter_cache.load_parameters_to_model(self.kernel, 0)
            del self.model

    def set_mean_function(self, constant: float):
        """
        setter function for mean function - sets the use_mean_function flag to True
        @TODO: mean function is not trainable right now - was only used for Safe Active learning

        Arguments:
            constant: constant value for mean function
        """
        self.use_mean_function = True
        self.mean_function = gpflow.mean_functions.Constant(c=constant)

    def set_number_of_restarts(self, n_starts: int):
        """
        Setter method to set the number of restarts in the multistart optimization - some models need more - default in the class is 10

        Arguments:
            n_starts: number of initial values in the multistart optimization
        """
        self.n_starts_for_multistart_opt = n_starts

    def infer(self, x_data: np.array, y_data: np.array):
        """
        Main entrance method for learning the model - training methods are called if hp should be done otherwise only gpflow model is build

        Arguments:
            x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
            y_data: Label array with shape (n,1) where n is the number of training points
        """
        self.build_model(x_data, y_data)
        if self.optimize_hps:
            if self.perform_multi_start_optimization:
                self.multi_start_optimization(self.n_starts_for_multistart_opt)
            else:
                self.optimize()

    def build_model(self, x_data: np.array, y_data: np.array):
        """
        Method to build to gpflow model object

        Arguments:
            x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
            y_data: Label array with shape (n,1) where n is the number of training points
        """
        assert len(y_data.shape) == 2
        if isinstance(self.kernel, InputInitializedKernelInterface):
            self.kernel.initialize_parameters(x_data, y_data)
            self.kernel_initial_parameter_cache.clear()
            self.kernel_initial_parameter_cache.store_parameters_from_model(self.kernel)

        if self.use_mean_function:
            self.model = gpflow.models.GPR(
                data=(x_data, y_data),
                kernel=self.kernel,
                mean_function=self.mean_function,
                noise_variance=np.power(self.observation_noise, 2.0),
            )
            set_trainable(self.model.mean_function.c, False)
        else:
            self.model = gpflow.models.GPR(
                data=(x_data, y_data),
                kernel=self.kernel,
                mean_function=None,
                noise_variance=np.power(self.observation_noise, 2.0),
            )
        set_trainable(self.model.likelihood.variance, self.train_likelihood_variance)
        if self.set_prior_on_observation_noise:
            self.model.likelihood.variance.prior = tfd.Exponential(1 / np.power(self.expected_observation_noise, 2.0))

    def optimize(self):
        """
        Method for performing Type-2 ML infernence - optimization is repeated if convergence was not succesfull or cholesky was not possible
        perturbation of initial values is applied in this case.
        If kernel parameters have prior this method automatically turns to MAP estimation!!
        """
        if self.sample_initial_parameters_at_start:
            self.sample_initial_parameters()

        if self.print_summaries:
            logger.debug("Initial parameters:")
            self.print_model_summary()

        optimizer = gpflow.optimizers.Scipy()
        optimization_success = False
        while not optimization_success:
            try:
                time_before_opt = time.perf_counter()
                opt_res = optimizer.minimize(self.training_loss, self.model.trainable_variables)
                time_after_opt = time.perf_counter()
                self.last_opt_time = time_after_opt - time_before_opt
                optimization_success = opt_res.success
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                logger.error("Error in optimization - try again")
                traceback.print_exc()
            if not optimization_success:
                logger.warning("Optimization failed - resample parameters and try again")
                self.sample_initial_parameters()
            else:
                logger.debug("Optimization successful - learned parameters:")
                if self.print_summaries:
                    self.print_model_summary()

    def get_last_opt_time(self):
        return self.last_opt_time

    def get_last_multistart_opt_time(self):
        return self.last_multi_start_opt_time

    def get_last_inference_time(self):
        if self.perform_multi_start_optimization:
            return self.get_last_multistart_opt_time()
        else:
            return self.get_last_opt_time()

    def training_loss(self) -> tf.Tensor:
        if self.add_kernel_hp_regularizer:
            return self.model.training_loss() + self.model.kernel.regularization_loss(self.model.data[0])
        else:
            return self.model.training_loss()

    def sample_initial_parameters(self):
        if self.initial_parameter_strategy == InitialParameters.PERTURB:
            self.pertube_parameters(self.perturbation_factor)
        elif self.initial_parameter_strategy == InitialParameters.UNIFORM_DISTRIBUTION:
            self.parameters_from_uniform_distribution()
        else:
            raise ValueError()

    def multi_start_optimization(self, n_starts: int):
        """
        Method for performing optimization (Type-2 ML) of kernel hps with multiple initial values - self.optimzation method is
        called multiple times and the log_posterior_density (falls back to log_marg_likeli for ML) is collected for all initial values.
        Model/kernel is set to the trained parameters with largest log_posterior_density

        Arguments:
            n_start: number of different initialization/restarts
        """
        optimization_success = False
        self.parameter_cache = GPParameterCache()
        while not optimization_success:
            self.parameter_cache.clear()
            assert len(self.parameter_cache.parameters_list) == 0
            assert len(self.parameter_cache.loss_list) == 0
            try:
                opt_time = 0.0
                self.multi_start_losses = []
                for i in range(0, n_starts):
                    logger.info(f"Optimization repeat {i + 1}/{n_starts}")
                    self.optimize()
                    after_optim = time.perf_counter()
                    loss = self.training_loss()
                    self.parameter_cache.store_parameters_from_model(self.model, loss, add_loss_value=True)
                    self.multi_start_losses.append(loss)
                    after_loss_calc = time.perf_counter()
                    opt_time += self.get_last_opt_time() + (after_loss_calc - after_optim)
                    if self.add_kernel_hp_regularizer:
                        logger.info(f"Loss for run: {loss}")
                    else:
                        log_marginal_likelihood = -1 * loss
                        logger.info(f"Log marginal likeli for run: {log_marginal_likelihood}")
                self.parameter_cache.load_best_parameters_to_model(self.model)
                self.last_multi_start_opt_time = opt_time
                logger.debug("Chosen parameter values:")
                self.print_model_summary()
                if self.add_kernel_hp_regularizer:
                    logger.debug(f"Loss of chosen parameters: {self.training_loss()}")
                else:
                    logger.debug(f"Marginal Likelihood of chosen parameters: {self.model.log_posterior_density()}")
                optimization_success = True
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                logger.error("Error in multistart optimization - repeat")

    def print_model_summary(self):
        if logger.isEnabledFor(logging.DEBUG):
            print_summary(self.model)

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> np.float:
        """
        Estimates the model evidence - always retrieves marg likelihood, also when HPs are provided with prior!!

        Arguments:
            x_data: (only necessary if infer was not yet called) Input array with shape (n,d) where d is the input dimension and n the number of training points
            y_data: (only necessary if infer was not yet called) Label array with shape (n,1) where n is the number of training points

        Returns:
            marginal likelihood value of infered model
        """
        if self.model is None and x_data is not None and y_data is not None:
            self.infer(x_data, y_data)
        model_evidence = self.model.log_marginal_likelihood().numpy()
        return model_evidence

    def pertube_parameters(self, factor_bound: float):
        """
        Method for perturbation of the current kernel parameters - internal method that is used before optimization

        Arguments:
            factor_bound: old value is mutliplied with (1+factor) where the factor is random and the factor_bound defines the interval of that variable
        """
        self.kernel_initial_parameter_cache.load_parameters_to_model(self.model.kernel, 0)
        if self.train_likelihood_variance:
            self.model.likelihood.variance.assign(np.power(self.observation_noise, 2.0))
        logger.debug("Pertube parameters - parameters before perturbation")
        self.print_model_summary()
        for variable in self.model.trainable_variables:
            unconstrained_value = variable.numpy()
            factor = 1 + np.random.uniform(-1 * factor_bound, factor_bound, size=unconstrained_value.shape)
            if np.isclose(unconstrained_value, 0.0, rtol=1e-07, atol=1e-09).all():
                new_unconstrained_value = (
                    unconstrained_value + np.random.normal(0, 0.05, size=unconstrained_value.shape)
                ) * factor
            else:
                new_unconstrained_value = unconstrained_value * factor
            variable.assign(new_unconstrained_value)

    def parameters_from_uniform_distribution(self):
        for parameter in self.model.trainable_parameters:
            shape = parameter.shape
            if parameter.name in ["chain_of_shift_of_softplus", "chain_of_shift_of_exp"]:
                sample = tfd.Uniform(low=1.000001e-06, high=self.initial_uniform_upper_bound).sample(sample_shape=shape)
            elif parameter.name == "softplus":
                sample = tfd.Uniform(low=1e-9, high=self.initial_uniform_upper_bound).sample(sample_shape=shape)
            else:
                sample = tfd.Uniform(
                    low=self.initial_uniform_lower_bound, high=self.initial_uniform_upper_bound
                ).sample(sample_shape=shape)
            parameter.assign(sample.numpy())

    def get_number_of_trainable_parameters(self):
        total_number = 0
        for variable in self.model.trainable_variables:
            total_number += tf.size(variable).numpy()
        return total_number

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        """
        Method for retrieving the predictive mean and sigma for a given array of the test points

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        mean array with shape (n,)
        sigma array with shape (n,)
        """
        if self.prediction_quantity == PredictionQuantity.PREDICT_F:
            pred_mus, pred_vars = self.model.predict_f(x_test)
        elif self.prediction_quantity == PredictionQuantity.PREDICT_Y:
            pred_mus, pred_vars = self.model.predict_y(x_test)
        pred_sigmas = np.sqrt(pred_vars)
        return np.squeeze(pred_mus), np.squeeze(pred_sigmas)

    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        """
        Method for calculating the log likelihood value of the the predictive distribution at the test input points (evaluated at the output values)
        - method is therefore for validation purposes only

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points
        y_test: Array of test output points with shape (n,1)

        Returns:
        array of shape (n,) with log liklihood values
        """
        pred_mus, pred_vars = self.model.predict_y(x_test)
        pred_sigmas = np.sqrt(pred_vars)
        log_likelis = norm.logpdf(np.squeeze(y_test), np.squeeze(pred_mus), np.squeeze(pred_sigmas))
        return log_likelis

    def predict_full_cov(self, x_test: np.array) -> Tuple[np.array, np.array]:
        """
        Method for retrieving the predictive mean and full covariance for a given array of the test points

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        mean array with shape (n,)
        sigma array with shape (n,n)
        """
        if self.prediction_quantity == PredictionQuantity.PREDICT_F:
            pred_mus, pred_cov = self.model.predict_f(x_test, full_cov=True)
        elif self.prediction_quantity == PredictionQuantity.PREDICT_Y:
            pred_mus, pred_cov = self.model.predict_y(x_test, full_cov=True)
        return np.squeeze(pred_mus), np.squeeze(pred_cov)

    def entropy_predictive_dist_full_cov(self, x_test: np.array) -> np.float:
        pred_mus, pred_cov = self.predict_full_cov(x_test)
        entropy = multivariate_normal(pred_mus, pred_cov).entropy()
        return entropy

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        """
        Method for calculating the entropy of the predictive distribution for test sequence - used for acquistion function in active learning

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        entropy array with shape (n,1)
        """
        if self.prediction_quantity == PredictionQuantity.PREDICT_F:
            pred_mus, pred_vars = self.model.predict_f(x_test)
        elif self.prediction_quantity == PredictionQuantity.PREDICT_Y:
            pred_mus, pred_vars = self.model.predict_y(x_test)
        pred_sigmas = np.sqrt(pred_vars)
        entropies = normal_entropy(pred_sigmas)
        return entropies

    def calculate_complete_information_gain(self, x_data: np.array) -> np.float:
        n = x_data.shape[0]
        gram_matrix = self.model.kernel.K(x_data)
        identity = np.identity(n)
        likelihood_variance = self.model.likelihood.variance.numpy()
        _, log_det = np.linalg.slogdet(identity + (1 / likelihood_variance) * gram_matrix)
        information_gain = 0.5 * log_det
        return information_gain

    def deactivate_summary_printing(self):
        self.print_summaries = False
