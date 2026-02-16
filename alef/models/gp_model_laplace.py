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

import logging
import traceback
from alef.kernels.input_initialized_kernel_interface import InputInitializedKernelInterface
from alef.oracles.exponential_2d import Exponential2D
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import gpflow
from gpflow.utilities import print_summary, set_trainable
import numpy as np


from enum import Enum
from typing import Tuple

from alef.kernels.regularized_kernel_interface import RegularizedKernelInterface
from alef.utils.utils import normal_entropy
from alef.utils.ground_truth_function import GroundTruthFunction, FunctionType
from alef.models.base_model import BaseModel
from alef.utils.gaussian_mixture_density_nd import GaussianMixtureDensityNd
from scipy.stats import multivariate_normal
from alef.models.batch_model_interface import BatchModelInterace

tf.executing_eagerly()

logger = logging.getLogger(__name__)


class PredictionType(Enum):
    SAMPLE_BASED = 1
    NORMAL_APPROXIMATION = 2
    MAP = 3


class GPModelLaplace(BaseModel, BatchModelInterace):
    def __init__(
        self,
        kernel: gpflow.kernels.Kernel,
        observation_noise: float,
        expected_observation_noise: float,
        train_likelihood_variance: bool,
        pertube_parameters_at_start: bool = False,
        perform_multi_start_optimization: bool = False,
        prediction_type: PredictionType = PredictionType.NORMAL_APPROXIMATION,
        n_starts_for_multistart_opt: int = 10,
        pertubation_for_multistart_opt: float = 0.5,
        **kwargs,
    ):
        self.kernel = gpflow.utilities.deepcopy(kernel)
        self.kernel_copy = gpflow.utilities.deepcopy(kernel)
        assert not isinstance(self.kernel, InputInitializedKernelInterface)  # not yet implemented
        self.observation_noise = observation_noise
        self.expected_observation_noise = expected_observation_noise
        self.train_likelihood_variance = train_likelihood_variance
        self.model = None
        self.prediction_type = prediction_type
        self.use_mean_function = False
        self.pertube_parameters_at_start = pertube_parameters_at_start
        self.perform_multi_start_optimization = perform_multi_start_optimization
        self.n_starts_for_multistart_opt = n_starts_for_multistart_opt
        self.pertubation_for_multistart_opt = pertubation_for_multistart_opt
        if isinstance(kernel, RegularizedKernelInterface):
            self.add_kernel_hp_regularizer = True
        else:
            self.add_kernel_hp_regularizer = False

    def reset_model(self):
        if self.model is not None:
            self.kernel = gpflow.utilities.deepcopy(self.kernel_copy)
            del self.model

    def build_model(self, x_data: np.array, y_data: np.array):
        assert len(y_data.shape) == 2
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

        if self.train_likelihood_variance:
            self.model.likelihood.variance.prior = tfd.Exponential(1.0 / np.power(self.expected_observation_noise, 2.0))
        else:
            set_trainable(self.model.likelihood.variance, False)

    def map_inference(self):
        if self.perform_multi_start_optimization:
            self.multi_start_optimization(self.n_starts_for_multistart_opt, self.pertubation_for_multistart_opt)
        else:
            self.optimize(self.pertube_parameters_at_start)
        self.map_variables = self.get_variable_numpy_values()

    def optimize(self, pertube_initial_parameters: bool, pertubation_factor=0.2):
        if pertube_initial_parameters:
            logger.debug("Initial parameters:")
            self.pertube_parameters(pertubation_factor)
            self.print_model_summary()
        optimizer = gpflow.optimizers.Scipy()
        optimization_success = False
        while not optimization_success:
            try:
                opt_res = optimizer.minimize(self.training_loss, self.model.trainable_variables)
                optimization_success = opt_res.success
            except:
                logger.error("Error in optimization - try again")
                traceback.print_exc()
                self.pertube_parameters(pertubation_factor)
            if not optimization_success:
                logger.warning("Not converged - try again")
                self.pertube_parameters(pertubation_factor)
            else:
                logger.info("Optimization succesful - learned parameters:")
                self.print_model_summary()

    def print_model_summary(self):
        if logger.isEnabledFor(logging.DEBUG):
            print_summary(self.model)

    def training_loss(self) -> tf.Tensor:
        if self.add_kernel_hp_regularizer:
            return self.model.training_loss() + self.model.kernel.regularization_loss(self.model.data[0])
        else:
            return self.model.training_loss()

    def log_posterior_density(self) -> tf.Tensor:
        if self.add_kernel_hp_regularizer:
            return self.model.log_posterior_density() - self.model.kernel.regularization_loss(self.model.data[0])
        else:
            return self.model.log_posterior_density()

    def multi_start_optimization(self, n_starts: int, pertubation_factor: float):
        if self.pertube_parameters_at_start:
            self.pertube_parameters(0.1)
        optimization_success = False
        while not optimization_success:
            try:
                parameters_over_runs = []
                log_posterior_densities = []
                for i in range(0, n_starts):
                    self.optimize(True, pertubation_factor)
                    parameter_values = self.get_parameter_numpy_values()
                    parameters_over_runs.append(parameter_values)
                    log_posterior_density = self.log_posterior_density()
                    logger.info("Log posterior for run:")
                    logger.info(log_posterior_density)
                    log_posterior_densities.append(log_posterior_density)
                best_run_index = np.argmax(np.array(log_posterior_densities))
                best_run_parameters = parameters_over_runs[best_run_index]
                self.set_parameters_to_values(best_run_parameters)
                logger.debug("Chosen parameter values:")
                self.print_model_summary()
                optimization_success = True
            except Exception as e:
                print(e)
                logger.error("Error in multistart optimization - repeat")

    def calculate_laplace_covariance(self):
        with tf.GradientTape(persistent=True) as t1:
            with tf.GradientTape(persistent=True) as t2:
                log_posterior = self.log_posterior_density()
            posterior_gradients = t2.gradient(log_posterior, self.model.trainable_variables)
            posterior_gradients_list = []
            for gradient in posterior_gradients:
                if len(gradient.shape) == 0:
                    posterior_gradients_list.append(tf.expand_dims(gradient, axis=0))
                else:
                    posterior_gradients_list.append(gradient)
            stacked_gradients = tf.concat(posterior_gradients_list, axis=0)
        posterior_hessian = t1.jacobian(stacked_gradients, self.model.trainable_variables, experimental_use_pfor=False)
        posterior_hessian_list = []
        for single_hessian in posterior_hessian:
            if len(single_hessian.shape) == 1:
                posterior_hessian_list.append(tf.expand_dims(single_hessian, axis=1))
            else:
                posterior_hessian_list.append(single_hessian)
        stacked_hessian = tf.concat(posterior_hessian_list, axis=1)
        self.hessian = stacked_hessian.numpy()
        self.posterior_covariance_matrix = np.linalg.inv(-1 * self.hessian)
        logger.info("-Posterior covariance approximated with Laplace")

    def infer(self, x_data: np.array, y_data: np.array):
        self.build_model(x_data, y_data)
        self.map_inference()
        self.calculate_laplace_covariance()
        logger.debug("Estimated untransformed parameters - MAP:")
        logger.debug(self.map_variables)
        logger.debug("Covariance matrix:")
        logger.debug(self.posterior_covariance_matrix)

    def estimate_model_evidence(self, x_data=None, y_data=None):
        if self.model is None and x_data is not None and y_data is not None:
            self.infer(x_data, y_data)
        unnormalized_posterior = np.exp(self.log_posterior_density())
        n_dim_param = self.posterior_covariance_matrix.shape[0]
        model_evidence = (
            np.sqrt(np.power(2 * np.pi, n_dim_param))
            * np.sqrt(np.linalg.det(self.posterior_covariance_matrix))
            * unnormalized_posterior
        )
        logger.info("Model evidence: " + str(model_evidence))
        return model_evidence

    def estimate_log_model_evidence(self, x_data=None, y_data=None):
        if self.model is None and x_data is not None and y_data is not None:
            self.infer(x_data, y_data)
        log_posterior_density = self.log_posterior_density()
        n_dim_param = self.posterior_covariance_matrix.shape[0]
        log_model_evidence = (
            log_posterior_density
            - 0.5 * np.linalg.slogdet(-1 * self.hessian)[1]
            + n_dim_param * 0.5 * np.math.log(2 * np.pi)
        )
        return log_model_evidence

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        if self.prediction_type == PredictionType.NORMAL_APPROXIMATION:
            # Marginalized - with approximation from Garnett et. al (2015)
            mus_f, pred_sigma_f = self.predictive_dist_normal_approx(x_test)
        elif self.prediction_type == PredictionType.SAMPLE_BASED:
            # Marginalized - sample based with samples from laplace normal
            mus_f, pred_sigma_f = self.predictive_dist_sample_based(x_test)
        elif self.prediction_type == PredictionType.MAP:
            # Not marginalized over hp posterior
            mus_f, pred_sigma_f = self.predictive_dist_f_map(x_test)
        return mus_f, pred_sigma_f

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        mus_f, pred_sigma_f = self.predictive_dist_normal_approx(x_test)
        entropies = normal_entropy(pred_sigma_f)
        return entropies

    def predictive_dist_sample_based(self, x_test: np.array, n_samples=500) -> Tuple[np.array, np.array]:
        pred_mus_over_samples = []
        pred_sigmas_over_samples = []
        hp_posterior = tfd.MultivariateNormalFullCovariance(
            np.concatenate(self.map_variables), self.posterior_covariance_matrix
        )
        for j in range(0, n_samples):
            posterior_sample = hp_posterior.sample()
            self.set_variables_from_unfolded_array(posterior_sample)
            pred_mus, pred_vars = self.model.predict_f(x_test)
            pred_sigmas = np.sqrt(pred_vars)
            pred_mus_over_samples.append(pred_mus)
            pred_sigmas_over_samples.append(pred_sigmas)
        self.set_variables_to_values(self.map_variables)
        pred_mus_complete = np.array(pred_mus_over_samples)
        pred_sigmas_complete = np.array(pred_sigmas_over_samples)
        n = x_test.shape[0]
        mus_over_inputs = []
        sigmas_over_inputs = []
        for i in range(0, n):
            mu = np.mean(pred_mus_complete[:, i])
            var = np.mean(
                np.power(pred_mus_complete[:, i], 2.0) + np.power(pred_sigmas_complete[:, i], 2.0) - np.power(mu, 2.0)
            )
            mus_over_inputs.append(mu)
            sigmas_over_inputs.append(np.sqrt(var))
        return np.array(mus_over_inputs), np.array(sigmas_over_inputs)

    def predictive_dist_f_map(self, x_test):
        self.set_variables_to_values(self.map_variables)
        pred_mus, pred_vars = self.model.predict_f(x_test)
        pred_sigmas = np.sqrt(pred_vars)
        return np.squeeze(pred_mus), np.squeeze(pred_sigmas)

    def predictive_dist_normal_approx(self, x_test):
        logger.info("-Predict")
        mus_f_map, sigma_f_map = self.predictive_dist_f_map(x_test)
        vars_f_map = np.power(sigma_f_map, 2.0)
        n = x_test.shape[0]
        logger.debug("-Calculate mean and var gradients")
        mean_gradients, var_gradients = self.mean_and_var_gradients(x_test)
        marginal_sigmas = []
        logger.debug("-Calculation done")
        for i in range(0, n):
            marginal_var = self.calculate_approx_marginal_pred_variance_at_single_point(
                vars_f_map[i], mean_gradients[i], var_gradients[i]
            )
            marginal_sigma = np.sqrt(marginal_var)
            marginal_sigmas.append(marginal_sigma)
        pred_sigma = np.array(marginal_sigmas)
        logger.info("-Prediction done")
        return mus_f_map, pred_sigma

    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        raise NotImplementedError

    def calculate_approx_marginal_pred_variance_at_single_point(self, var_f_map, mean_gradient, var_gradient):
        variance = (
            4 / 3 * var_f_map
            + np.matmul(np.matmul(mean_gradient, self.posterior_covariance_matrix), mean_gradient)
            + (1 / (3 * var_f_map)) * np.matmul(np.matmul(var_gradient, self.posterior_covariance_matrix), var_gradient)
        )
        return variance

    def mean_and_var_gradients(self, x_grid):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            mean_tensor, var_tensor = self.model.predict_f(x_grid)
            mean_tensor_squeezed = tf.squeeze(mean_tensor)
            var_tensor_squeezed = tf.squeeze(var_tensor)
        mean_gradients_tuple = tape.jacobian(
            mean_tensor_squeezed, self.model.trainable_variables, experimental_use_pfor=False
        )
        var_gradient_tuple = tape.jacobian(
            var_tensor_squeezed, self.model.trainable_variables, experimental_use_pfor=False
        )
        mean_gradients_list = []
        var_gradients_list = []
        for mean_gradient in mean_gradients_tuple:
            if len(mean_gradient.shape) == 1:
                mean_gradients_list.append(tf.expand_dims(mean_gradient, axis=1))
            else:
                mean_gradients_list.append(mean_gradient)

        for var_gradient in var_gradient_tuple:
            if len(var_gradient.shape) == 1:
                var_gradients_list.append(tf.expand_dims(var_gradient, axis=1))
            else:
                var_gradients_list.append(var_gradient)

        mean_gradients = tf.concat(mean_gradients_list, axis=1).numpy()
        var_gradients = tf.concat(var_gradients_list, axis=1).numpy()
        return mean_gradients, var_gradients

    ########### Batch Calculation #################

    def predict_full_cov(self, x_test: np.array) -> Tuple[np.array, np.array]:
        mean_map, cov_map = self.model.predict_f(x_test, full_cov=True)
        mean_gradient, cov_gradients = self.mean_and_cov_gradients(x_test)
        marg_cov = self.calulate_approx_marginal_pred_covariance_at_single_point(
            np.squeeze(cov_map.numpy()), mean_gradient, cov_gradients
        )
        return np.squeeze(mean_map.numpy()), marg_cov, np.squeeze(cov_map.numpy())

    def mean_and_cov_gradients(self, x_test):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            mean_tensor, cov_tensor = self.model.predict_f(x_test, full_cov=True)
            mean_tensor_squeezed = tf.squeeze(mean_tensor, axis=-1)
            cov_tensor_squeezed = tf.squeeze(cov_tensor, axis=0)

        mean_gradients_tuple = tape.jacobian(
            mean_tensor_squeezed, self.model.trainable_variables, experimental_use_pfor=False
        )
        cov_gradient_tuple = tape.jacobian(
            cov_tensor_squeezed, self.model.trainable_variables, experimental_use_pfor=False
        )

        mean_gradients_list = []
        cov_gradients_list = []
        for mean_gradient in mean_gradients_tuple:
            if len(mean_gradient.shape) == 1:
                mean_gradients_list.append(tf.expand_dims(mean_gradient, axis=1))
            else:
                mean_gradients_list.append(mean_gradient)

        for cov_gradient in cov_gradient_tuple:
            if len(cov_gradient.shape) == 2:
                cov_gradients_list.append(tf.expand_dims(cov_gradient, axis=2))
            else:
                cov_gradients_list.append(cov_gradient)

        mean_gradients = tf.concat(mean_gradients_list, axis=1).numpy()
        cov_gradients = tf.concat(cov_gradients_list, axis=2).numpy()
        return mean_gradients, cov_gradients

    def calulate_approx_marginal_pred_covariance_at_single_point(self, cov_f_map, mean_gradient, cov_gradients):
        inv_cov_map = np.linalg.inv(cov_f_map)
        n_param = cov_gradients.shape[2]
        vars = np.diag(cov_f_map)
        assert n_param == self.posterior_covariance_matrix.shape[0]
        eval_points = [-1 * np.expand_dims(np.sqrt(vars / 3), axis=1)]
        marg_covs = []
        for point in eval_points:
            a_s = []
            for i in range(0, n_param):
                cov_gradient_for_parameter = cov_gradients[:, :, i]
                mean_gradient_for_parameter = np.expand_dims(mean_gradient[:, i], axis=1)
                a_paramter = mean_gradient_for_parameter + np.matmul(
                    np.matmul(cov_gradient_for_parameter, inv_cov_map), point
                )
                a_s.append(a_paramter)
            A = np.concatenate(a_s, axis=1)
            marg_cov = cov_f_map + np.matmul(np.matmul(A, self.posterior_covariance_matrix), A.T)
            marg_covs.append(marg_cov)
        gmm = GaussianMixtureDensityNd(
            np.repeat(1 / len(eval_points), len(eval_points)),
            np.array([np.zeros(vars.shape)] * len(eval_points)),
            np.array(marg_covs),
        )
        marg_cov_matched = gmm.covariance_matrix()
        return marg_cov_matched

    def hyper_batch_bald(self, x_test):
        mean_map, cov_map = self.model.predict_f(x_test, full_cov=True)
        mean_gradient, cov_gradients = self.mean_and_cov_gradients(x_test)
        marg_cov = self.calulate_approx_marginal_pred_covariance_at_single_point(
            np.squeeze(cov_map.numpy(), axis=0), mean_gradient, cov_gradients
        )
        cov_map = np.squeeze(cov_map.numpy(), axis=0)
        mean_map = np.squeeze(mean_map.numpy(), axis=-1)
        marg_entropy = multivariate_normal(mean_map, marg_cov).entropy()
        map_entropy = multivariate_normal(mean_map, cov_map).entropy()
        return marg_entropy - map_entropy

    def entropy_predictive_dist_full_cov(self, x_test: np.array) -> np.float:
        mean_map, cov_map = self.model.predict_f(x_test, full_cov=True)
        mean_gradient, cov_gradients = self.mean_and_cov_gradients(x_test)
        marg_cov = self.calulate_approx_marginal_pred_covariance_at_single_point(
            np.squeeze(cov_map.numpy(), axis=0), mean_gradient, cov_gradients
        )
        mean_map = np.squeeze(mean_map.numpy(), axis=-1)
        marg_entropy = multivariate_normal(mean_map, marg_cov).entropy()
        return marg_entropy

    ########### Utils/Getter/Setter ###############

    def hyper_bald(self, x_test, use_untransformed_information_gain=True):
        mus_f_map, sigma_f_map = self.predictive_dist_f_map(x_test)
        vars_f_map = np.power(sigma_f_map, 2.0)
        n = x_test.shape[0]
        mean_gradients, var_gradients = self.mean_and_var_gradients(x_test)
        information_gains = []
        marginal_entropies = []
        map_entropies = []
        for i in range(0, n):
            marginal_var = self.calculate_approx_marginal_pred_variance_at_single_point(
                vars_f_map[i], mean_gradients[i], var_gradients[i]
            )
            if use_untransformed_information_gain:
                marginal_entropy = 0.5 * np.log(2 * np.pi * np.exp(1.0) * marginal_var)
                map_entropy = 0.5 * np.log(2 * np.pi * np.exp(1.0) * vars_f_map[i])
                information_gain = marginal_entropy - map_entropy
            else:
                information_gain = marginal_var / vars_f_map[i]
            information_gains.append(information_gain)
            marginal_entropies.append(marginal_entropy)
            map_entropies.append(map_entropy)
        return np.array(information_gains)

    def get_parameter_numpy_values(self):
        parameter_values = []
        for parameter in self.model.trainable_parameters:
            parameter_value = parameter.numpy()
            parameter_values.append(parameter_value)
        return parameter_values

    def get_variable_numpy_values(self):
        variable_values = []
        for variable in self.model.trainable_variables:
            variable_value = variable.numpy()
            variable_values.append(variable_value)
        return variable_values

    def set_parameters_to_values(self, parameter_values):
        counter = 0
        for parameter in self.model.trainable_parameters:
            if parameter is self.model.likelihood.variance:
                if parameter_values[counter] < 1.000001e-06:
                    parameter.assign(1.000001e-06)
                    logger.info("Likelihood variance close to 1e-06 - set to 1.000001e-06")
                else:
                    parameter.assign(parameter_values[counter])
            else:
                parameter.assign(parameter_values[counter])
            counter += 1

    def set_variables_to_values(self, variable_values):
        counter = 0
        for variable in self.model.trainable_variables:
            variable.assign(variable_values[counter])
            counter += 1

    def set_variables_from_unfolded_array(self, unfolded_variable_array):
        assert len(unfolded_variable_array.shape) == 1
        index = 0
        new_variables_values = []
        for variable in self.model.trainable_variables:
            variable_value = variable.numpy()
            for i in range(0, variable_value.shape[0]):
                variable_value[i] = unfolded_variable_array[index]
                index += 1
            new_variables_values.append(variable_value)
        self.set_variables_to_values(new_variables_values)

    def check_kernel_parameter_shapes(self):
        for parameter in self.kernel.trainable_parameters:
            assert len(parameter.numpy().shape) == 1

    def set_mean_function(self, constant):
        self.use_mean_function = True
        self.mean_function = gpflow.mean_functions.Constant(c=constant)

    def pertube_parameters(self, factor_bound):
        self.model.kernel = gpflow.utilities.deepcopy(self.kernel_copy)
        if self.train_likelihood_variance:
            self.model.likelihood.variance.assign(np.power(self.observation_noise, 2.0))
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


if __name__ == "__main__":
    gt_function = Exponential2D(0.01)
    x_data, y_data = gt_function.get_random_data(40)
    # print(y_data.shape)
    gt_function = GroundTruthFunction(FunctionType.SIN_LIN, 0.1)
    # x_data,y_data = gt_function.produce_data(0,10,10)
    # x_data = np.expand_dims(x_data,axis=1)
    hyper_model = GPModelLaplace(gpflow.kernels.RBF(lengthscales=[1.0, 1.0], variance=[1.0]), 0.01)
    x_test = np.array([[1.0, 0.5], [0.0, 0.0]])
    hyper_model.infer(x_data, np.expand_dims(y_data, axis=1))
    hyper_model.predict_full_cov(x_test)
