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

from typing import Callable, Optional, Tuple
import numpy as np
from alef.kernels.input_initialized_kernel_interface import InputInitializedKernelInterface
from alef.kernels.regularized_kernel_interface import RegularizedKernelInterface
from alef.models.base_model import BaseModel
from enum import Enum
import gpflow
from gpflow.utilities import set_trainable
from tensorflow_probability import distributions as tfd
from alef.utils.gp_paramater_cache import GPParameterCache
from alef.utils.utils import k_means, normal_entropy
import tensorflow as tf
from alef.utils.utils import calculate_rmse
from scipy.stats import norm
from alef.kernels.deep_kernels.base_deep_kernel import BaseDeepKernel
from alef.enums.global_model_enums import PredictionQuantity


class BaseGPType(Enum):
    GPR = 1
    SGPR = 2
    SVGP = 3


class OptimizerType(Enum):
    ADAM = 1


class ValidationMetric(Enum):
    RMSE = 0
    NLL = 1


class GPModelScalable(BaseModel):
    """
    This class enables training of GP methods like standard Deep methods in the following ways:
    - trainable parameters are learned with Adam (first order gradient methods)
    - a validation set can be used - parameter values are used with best val performance over training
    - @TODO: mini-batching can be used (only for SVGP model)
    One can also specify which GP approx might be used (full GP (GPR), Sparse GP (SPGR) or Sparse Variational GP(SVGP))
    """

    def __init__(
        self,
        kernel: gpflow.kernels.Kernel,
        base_gp_type: BaseGPType,
        optimizer_type: OptimizerType,
        n_inducing_points: int,
        n_iterations: int,
        n_repeats: int,
        learning_rate: float,
        use_mini_batches: bool,
        use_validation_set: bool,
        val_fraction: float,
        validation_metric: ValidationMetric,
        initial_observation_noise: float,
        expected_observation_noise: float,
        set_prior_on_observation_noise: bool,
        **kwargs,
    ):
        self.kernel = kernel
        self.base_gp_type = base_gp_type
        self.optimizer_type = optimizer_type
        self.use_mini_batches = use_mini_batches
        self.use_validation_set = use_validation_set
        self.initial_observation_noise = initial_observation_noise
        self.expected_observation_noise = expected_observation_noise
        self.set_prior_on_observation_noise = set_prior_on_observation_noise
        self.n_inducing_points = n_inducing_points
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.prediction_quantity = PredictionQuantity.PREDICT_Y
        self.validation_metric = validation_metric
        self.val_fraction = val_fraction
        self.n_repeats = n_repeats
        self.initial_parameter_cache = None
        if isinstance(kernel, RegularizedKernelInterface):
            self.add_kernel_hp_regularizer = True
        else:
            self.add_kernel_hp_regularizer = False
        self.initialize_optimizer()

    def initialize_optimizer(self):
        if self.optimizer_type == OptimizerType.ADAM:
            self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

    def build_model(self, x_train, y_train):
        n_train = x_train.shape[0]
        if self.base_gp_type == BaseGPType.SGPR or self.base_gp_type == BaseGPType.SVGP:
            if n_train > self.n_inducing_points:
                inducing_locations = k_means(self.n_inducing_points, x_train)
            else:
                inducing_locations = x_train.copy()

        if self.base_gp_type == BaseGPType.GPR:
            self.model = gpflow.models.GPR(
                data=(x_train, y_train),
                kernel=self.kernel,
                mean_function=None,
                noise_variance=np.power(self.initial_observation_noise, 2.0),
            )
        elif self.base_gp_type == BaseGPType.SGPR:
            self.model = gpflow.models.SGPR(
                data=(x_train, y_train),
                kernel=self.kernel,
                inducing_variable=inducing_locations,
                mean_function=None,
                noise_variance=np.power(self.initial_observation_noise, 2.0),
            )
            set_trainable(self.model.inducing_variable, False)
        elif self.base_gp_type == BaseGPType.SVGP:
            self.model = gpflow.models.SVGP(
                kernel=self.kernel,
                likelihood=gpflow.likelihoods.Gaussian(variance=np.power(self.initial_observation_noise, 2.0)),
                inducing_variable=inducing_locations,
                num_data=n_train,
            )
            set_trainable(self.model.inducing_variable, False)
        if self.set_prior_on_observation_noise:
            self.model.likelihood.variance.prior = tfd.Exponential(1 / np.power(self.expected_observation_noise, 2.0))

    def infer(self, x_data: np.array, y_data: np.array):
        if self.use_validation_set:
            x_train, y_train, x_val, y_val = self.train_val_split(x_data, y_data)
        else:
            x_train = x_data
            y_train = y_data
            x_val = None
            y_val = None
        if isinstance(self.kernel, InputInitializedKernelInterface):
            self.kernel.initialize_parameters(x_train, y_train)
        self.build_model(x_train, y_train)
        self.switch_to_train_mode()
        if self.initial_parameter_cache is None:
            self.initial_parameter_cache = GPParameterCache()
            self.initial_parameter_cache.store_parameters_from_model(self.model)
        self.parameter_cache = GPParameterCache()
        for i in range(0, self.n_repeats):
            self.initial_parameter_cache.load_parameters_to_model(self.model, 0)
            if self.use_mini_batches:
                assert self.base_gp_type == BaseGPType.SVGP
                training_loss, val_metric = self.optimize_with_mini_batch(x_train, y_train, x_val, y_val)
            else:
                training_loss, val_metric = self.optimize(x_train, y_train, x_val, y_val)
            if self.use_validation_set:
                self.parameter_cache.store_parameters_from_model(self.model, val_metric, add_loss_value=True)
            else:
                self.parameter_cache.store_parameters_from_model(self.model, training_loss, add_loss_value=True)
        self.parameter_cache.load_best_parameters_to_model(self.model)
        self.switch_to_inference_mode()
        print("Loss final model:")
        loss = self.loss(x_train, y_train)()
        print(loss)

    def switch_to_train_mode(self):
        if isinstance(self.model.kernel, BaseDeepKernel):
            self.model.kernel.set_mode(True)

    def switch_to_inference_mode(self):
        if isinstance(self.model.kernel, BaseDeepKernel):
            self.model.kernel.set_mode(False)

    def loss(self, x_train, y_train):
        if self.add_kernel_hp_regularizer:
            base_training_loss = self.base_loss_closure(x_train, y_train)
            regularization_loss = self.kernel_regularization_loss_closure(x_train)
            training_loss = lambda: base_training_loss() + regularization_loss()
        else:
            training_loss = self.base_loss_closure(x_train, y_train)
        return training_loss

    def base_loss_closure(self, x_batch: Optional[np.array] = None, y_batch: Optional[np.array] = None) -> Callable:
        if self.base_gp_type == BaseGPType.GPR or self.base_gp_type == BaseGPType.SGPR:
            loss_func = self.model.training_loss_closure()
        elif self.base_gp_type == BaseGPType.SVGP:
            assert not x_batch is None
            assert not y_batch is None
            loss_func = self.model.training_loss_closure((x_batch, y_batch))
        return loss_func

    def kernel_regularization_loss_closure(self, x_data):
        @tf.function
        def kernel_regularization_loss():
            return self.model.kernel.regularization_loss(x_data)

        return kernel_regularization_loss

    def train_val_split(self, x_data: np.array, y_data: np.array) -> Tuple[np.array, np.array, np.array, np.array]:
        len_data = x_data.shape[0]
        len_val = int(len_data * self.val_fraction)
        len_train = len_data - len_val
        x_train = x_data[:len_train]
        y_train = y_data[:len_train]
        x_val = x_data[len_train:]
        y_val = y_data[len_train:]
        assert x_val.shape[0] == len_val
        assert x_train.shape[0] == len_train
        return x_train, y_train, x_val, y_val

    def calc_validation_metric(self, x_val, y_val):
        if self.validation_metric == ValidationMetric.RMSE:
            pred_mu_val, _ = self.predictive_dist(x_val)
            val_metric = calculate_rmse(pred_mu_val, y_val)
            return val_metric
        elif self.validation_metric == ValidationMetric.NLL:
            val_metric = np.mean(-1 * self.predictive_log_likelihood(x_val, y_val))
            return val_metric

    def optimize(
        self, x_train: np.array, y_train: np.array, x_val: Optional[np.array] = None, y_val: Optional[np.array] = None
    ):
        print("-Start optimization")
        training_loss = self.loss(x_train, y_train)
        local_parameter_cache = GPParameterCache()
        for step in range(0, self.n_iterations):
            try:
                self.switch_to_train_mode()
                self.optimizer.minimize(training_loss, self.model.trainable_variables)
                loss_value = training_loss().numpy()
            except Exception:
                print("-Error occured in minimization step")

            if step % 10 == 0 and step > 0:
                print(f"-Step: {step}/{self.n_iterations}")
                pred_mu, _ = self.predictive_dist(x_train)
                train_rmse = calculate_rmse(pred_mu, y_train)
                if self.use_validation_set:
                    val_metric = self.calc_validation_metric(x_val, y_val)
                    local_parameter_cache.store_parameters_from_model(self.model, val_metric, True)
                    print(f"Loss: {loss_value} Train RMSE: {train_rmse} Val Metric: {val_metric}")
                else:
                    print(f"Loss: {loss_value} Train RMSE: {train_rmse}")
                # print_summary(self.model)
        if self.use_validation_set:
            print(local_parameter_cache.loss_list)
            local_parameter_cache.load_best_parameters_to_model(self.model)
            val_metric_final = self.calc_validation_metric(x_val, y_val)
            print(val_metric_final)
        else:
            val_metric_final = None

        self.switch_to_inference_mode()
        training_loss_final = training_loss().numpy()
        print(training_loss_final)
        print("-Optimization finished")
        return training_loss_final, val_metric_final

    def optimize_with_mini_batch(
        self, x_train: np.array, y_train: np.array, x_val: Optional[np.array] = None, y_val: Optional[np.array] = None
    ):
        # @TODO Implement mini batching
        raise NotImplementedError

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        self.switch_to_inference_mode()
        if self.prediction_quantity == PredictionQuantity.PREDICT_F:
            pred_mus, pred_vars = self.model.predict_f(x_test)
        elif self.prediction_quantity == PredictionQuantity.PREDICT_Y:
            pred_mus, pred_vars = self.model.predict_y(x_test)
        pred_sigmas = np.sqrt(pred_vars)
        return np.squeeze(pred_mus), np.squeeze(pred_sigmas)

    def reset_model(self):
        pass

    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        self.switch_to_inference_mode()
        pred_mus, pred_vars = self.model.predict_y(x_test)
        pred_sigmas = np.sqrt(pred_vars)
        log_likelis = norm.logpdf(np.squeeze(y_test), np.squeeze(pred_mus), np.squeeze(pred_sigmas))
        return log_likelis

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        self.switch_to_inference_mode()
        if self.prediction_quantity == PredictionQuantity.PREDICT_F:
            pred_mus, pred_vars = self.model.predict_f(x_test)
        elif self.prediction_quantity == PredictionQuantity.PREDICT_Y:
            pred_mus, pred_vars = self.model.predict_y(x_test)
        pred_sigmas = np.sqrt(pred_vars)
        entropies = normal_entropy(pred_sigmas)
        return entropies

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> np.float:
        pass
