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

from enum import Enum
import numpy as np
import time
import traceback
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import gpflow
from gpflow.utilities import print_summary, set_trainable
from scipy.stats import norm

from alef.utils.gp_paramater_cache import GPParameterCache
from alef.models.base_model import BaseModel
from alef.models.mo_gpr_transfer import TransferGPR
from alef.kernels.multi_output_kernels.base_transfer_kernel import BaseTransferKernel
from alef.kernels.regularized_kernel_interface import RegularizedKernelInterface
from alef.enums.global_model_enums import PredictionQuantity, InitialParameters
import logging

logger = logging.getLogger(__name__)


class SourceTraining(Enum):
    WITHOUT_TARGET = 0
    WITH_TARGET = 1


class TransferGPModel(BaseModel):
    """
    Class that implements multi-output Gaussian process regression with Type-2 ML for kernel hyperparameter infernece. It forms mainly a wrapper around
    the MOGPR object

    Attributes:
        kernel: kernel that is used inside the Gaussian process - needs to be child of gpflow.kernels.MultioutputKernel
        model: holds the MOGPR instance
        optimize_hps: bool if kernel parameters are trained
        train_likelihood_variance: bool if likelihood variance is trained
        observation_noise: observation noise level - is either set fixed to that value or acts as initial starting value for optimization
        pertube_parameters_at_start: bool if parameters of the kernels should be pertubed before optimization
        set_prior_on_observation_noise: bool if prior should be applied to obvservation noise (Exponential prior with expected value self.observation_noise)
    """

    def __init__(
        self,
        kernel: BaseTransferKernel,
        observation_noise: float,
        expected_observation_noise: float,
        optimize_hps: bool,
        train_likelihood_variance: bool,
        sample_initial_parameters_at_start: bool,
        initial_parameter_strategy: InitialParameters,
        perturbation_for_multistart_opt: float = 0.5,
        perturbation_for_singlestart_opt: float = 0.1,
        source_training_mode: SourceTraining = SourceTraining.WITHOUT_TARGET,
        perform_multi_start_optimization=False,
        n_starts_for_multistart_opt: int = 5,
        set_prior_on_observation_noise=False,
        prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_F,
        **kwargs,
    ):
        self.kernel = kernel
        self.kernel_copy = gpflow.utilities.deepcopy(kernel)
        self.source_trained = False
        self.source_data = None
        self.kernel_on_source = None
        self.likelihood_vars_on_source = None
        self.cholesky_on_source = None
        self.observation_noise = observation_noise
        self.expected_observation_noise = expected_observation_noise
        self.model = None
        self.optimize_hps = optimize_hps
        self.train_likelihood_variance = train_likelihood_variance
        self.source_training_mode = source_training_mode
        self.use_mean_function = False
        self.sample_initial_parameters_at_start = sample_initial_parameters_at_start
        self.initial_parameter_strategy = initial_parameter_strategy
        self.perform_multi_start_optimization = perform_multi_start_optimization

        if self.perform_multi_start_optimization:
            self.perturbation_factor = perturbation_for_multistart_opt
        else:
            self.perturbation_factor = perturbation_for_singlestart_opt

        self.n_starts_for_multistart_opt = n_starts_for_multistart_opt
        self.set_prior_on_observation_noise = set_prior_on_observation_noise
        self.prediction_quantity = prediction_quantity
        self.print_summaries = False
        if isinstance(kernel, RegularizedKernelInterface):
            self.add_kernel_hp_regularizer = True
        else:
            self.add_kernel_hp_regularizer = False

    def assign_likelihood_variance(self):
        new_value = np.power(self.observation_noise, 2.0)
        t = self.kernel.output_dimension
        if hasattr(new_value, "__len__"):
            assert len(new_value) == t
            if self.source_trained:
                new_value[:-1] = np.array(self.likelihood_vars_on_source[:-1])
        else:
            new_value = new_value * np.ones(t, dtype=float)
            if self.source_trained:
                new_value[:-1] = np.array(self.likelihood_vars_on_source[:-1])

        for i, lik in enumerate(self.model.likelihood.likelihoods):
            lik.variance.assign(new_value[i])

    def reset_model(self, reset_source: bool = False):
        """
        resets the model to the initial values - kernel parameters and observation noise are reset to initial values - gpflow model is deleted
        """
        if self.model is not None:
            if reset_source:
                self.kernel = gpflow.utilities.deepcopy(self.kernel_copy)
                self.source_trained = False
                self.source_data = None
                del self.model
            elif self.source_trained:
                self.kernel = gpflow.utilities.deepcopy(self.kernel_on_source)
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

    def build_model(self, x_data: np.array, y_data: np.array):
        yy = np.hstack((y_data[..., :1], x_data[..., -1, None]))
        t = self.kernel.output_dimension - 1
        t_mask = x_data[:, -1] == t
        np.power(self.observation_noise, 2.0)
        model = TransferGPR
        if self.use_mean_function:
            self.model = model(
                source_data=(x_data[~t_mask], yy[~t_mask]),
                data=(x_data[t_mask], yy[t_mask]),
                kernel=self.kernel,
                mean_function=self.mean_function,
                noise_variance=np.power(self.observation_noise, 2.0),
            )
            set_trainable(self.model.mean_function.c, False)
        else:
            self.model = model(
                source_data=(x_data[~t_mask], yy[~t_mask]),
                data=(x_data[t_mask], yy[t_mask]),
                kernel=self.kernel,
                mean_function=None,
                noise_variance=np.power(self.observation_noise, 2.0),
            )

        if self.set_prior_on_observation_noise:
            for lik in self.model.likelihood.likelihoods:
                lik.variance.prior = tfd.Exponential(1 / np.power(self.expected_observation_noise, 2.0))

    def _set_source(self):
        t = self.kernel.output_dimension - 1

        self.model.kernel = gpflow.utilities.deepcopy(self.kernel_on_source)
        self.model.source_data = self.source_data

        for p, lik in enumerate(self.model.likelihood.likelihoods):
            if p == t:
                set_trainable(lik.variance, self.train_likelihood_variance)
            else:
                lik.variance.assign(self.likelihood_vars_on_source[p])
                set_trainable(lik.variance, False)
        self.likelihood_vars_on_source = [lik.variance.numpy() for lik in self.model.likelihood.likelihoods]
        self.model.set_source_cholesky(self.cholesky_on_source)

    def _infer_and_set_source(self):
        T = self.model.kernel.output_dimension - 1

        if self.optimize_hps:
            # set trainable
            for p, lik in enumerate(self.model.likelihood.likelihoods):
                if p == T and self.source_training_mode == SourceTraining.WITHOUT_TARGET:
                    set_trainable(lik.variance, False)
                else:
                    set_trainable(lik.variance, self.train_likelihood_variance)
            if self.source_training_mode == SourceTraining.WITHOUT_TARGET:
                target_trainable = self.model.kernel.get_target_parameters_trainable()
                self.model.kernel.set_target_parameters_trainable(False)

            # optimize
            if self.perform_multi_start_optimization:
                self.multi_start_optimization(self.n_starts_for_multistart_opt)
            else:
                self.optimize()

            # set trainable
            self.model.kernel.set_source_parameters_trainable(False)
            if self.source_training_mode == SourceTraining.WITHOUT_TARGET:
                self.model.kernel.set_target_parameters_trainable(target_trainable)
            for p, lik in enumerate(self.model.likelihood.likelihoods):
                if p == T:
                    set_trainable(lik.variance, self.train_likelihood_variance)
                else:
                    set_trainable(lik.variance, False)

            self.source_trained = True
            self.source_data = self.model.source_data
            self.kernel_on_source = gpflow.utilities.deepcopy(self.model.kernel)
            self.likelihood_vars_on_source = [lik.variance.numpy() for lik in self.model.likelihood.likelihoods]
            self.cholesky_on_source = self.model.compute_source_cholesky()

            self.model.set_source_cholesky(self.cholesky_on_source)

    def infer(self, x_data: np.array, y_data: np.array):
        """
        Main entrance method for learning the model - training methods are called if hp should be done otherwise only gpflow model is build

        Arguments:
            x_data: Input array with shape (n,d+1) where d is the input dimension and n the number of training points
            y_data: Label array with shape (n,1) where n is the number of training points
        """
        self.build_model(x_data, y_data)
        if self.source_trained:
            self._set_source()
            t_opt_source = 0
            if self.perform_multi_start_optimization:
                t_opt_multi_start_source = 0
        else:
            self._infer_and_set_source()
            t_opt_source = self.get_last_opt_time()
            if self.perform_multi_start_optimization:
                t_opt_multi_start_source = self.get_last_multistart_opt_time()
        if tf.shape(self.model.data[0])[0] == 0:
            print("no target data")
            return

        if self.optimize_hps:
            if self.perform_multi_start_optimization:
                self.multi_start_optimization(self.n_starts_for_multistart_opt)
                self.last_opt_time = t_opt_source + self.get_last_opt_time()
                self.last_multi_start_opt_time = t_opt_multi_start_source + self.get_last_multistart_opt_time()
            else:
                self.optimize()
                self.last_opt_time = t_opt_source + self.get_last_opt_time()

    def optimize(self):
        """
        Method for performing Type-2 ML infernence - optimization is repeated if convergence was not succesfull or cholesky was not possible
        pertubation of initial values is applied in this case.
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
                opt_res = optimizer.minimize(self.model.training_loss, self.model.trainable_variables)
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
                    print("Optimization succesful - learned parameters:")
                    print_summary(self.model)

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

    def print_model_summary(self):
        if logger.isEnabledFor(logging.DEBUG):
            print_summary(self.model)

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
        Method for pertubation of the current kernel parameters - internal method that is used before optimization

        Arguments:
            factor_bound: old value is mutliplied with (1+factor) where the factor is random and the factor_bound defines the interval of that variable
        """
        if self.source_trained:
            self._set_source()
        else:
            self.model.kernel = gpflow.utilities.deepcopy(self.kernel_copy)
            if self.train_likelihood_variance:
                obs_vars = np.power(self.observation_noise, 2.0)
                t = self.kernel.output_dimension
                if hasattr(obs_vars, "__len__"):
                    assert len(obs_vars) == t
                else:
                    obs_vars = obs_vars * np.ones(t, dtype=float)

                for i, lik in enumerate(self.model.likelihood.likelihoods):
                    lik.variance.assign(obs_vars[i])

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

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        """
        Method for retrieving the predictive mean and sigma for a given array of the test points

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        mean array with shape (n,m)
        sigma array with shape (n,m)
        """
        if self.prediction_quantity == PredictionQuantity.PREDICT_F:
            pred_mus, pred_vars = self.model.predict_f(x_test)
        elif self.prediction_quantity == PredictionQuantity.PREDICT_Y:
            pred_mus, pred_vars = self.model.predict_y(x_test)
        pred_sigmas = np.sqrt(pred_vars)
        return np.squeeze(pred_mus), np.squeeze(pred_sigmas)

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        """
        Method for calculating the entropy of the predictive distribution for test sequence - used for acquistion function in active learning

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        entropy array with shape (n,1)
        """
        _, cov = self.model.predict_f(x_test, full_output_cov=True)
        cov = cov.numpy()

        entropy = 0.5 * np.log((2 * np.pi * np.e) ** cov.shape[-1] * np.linalg.det(cov))
        entropy = entropy.reshape([-1, 1])

        return entropy

    def calculate_complete_information_gain(self, x_data: np.array) -> np.float:
        raise NotImplementedError

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
