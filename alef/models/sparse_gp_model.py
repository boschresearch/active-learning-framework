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

from alef.enums.global_model_enums import InitialParameters
from alef.kernels.input_initialized_kernel_interface import InputInitializedKernelInterface
from alef.models.gp_model import GPModel, PredictionQuantity
import gpflow
import numpy as np
from alef.utils.utils import k_means
from gpflow.utilities import set_trainable
from tensorflow_probability import distributions as tfd


class SparseGpModel(GPModel):
    def __init__(
        self,
        kernel: gpflow.kernels.Kernel,
        observation_noise: float,
        optimize_hps: bool,
        train_likelihood_variance: bool,
        n_inducing_points: int,
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
        super().__init__(
            kernel,
            observation_noise,
            optimize_hps,
            train_likelihood_variance,
            sample_initial_parameters_at_start=sample_initial_parameters_at_start,
            initial_parameter_strategy=initial_parameter_strategy,
            perturbation_for_multistart_opt=perturbation_for_multistart_opt,
            perturbation_for_singlestart_opt=perturbation_for_singlestart_opt,
            initial_uniform_lower_bound=initial_uniform_lower_bound,
            initial_uniform_upper_bound=initial_uniform_upper_bound,
            perform_multi_start_optimization=perform_multi_start_optimization,
            set_prior_on_observation_noise=set_prior_on_observation_noise,
            n_starts_for_multistart_opt=n_starts_for_multistart_opt,
            expected_observation_noise=expected_observation_noise,
            prediction_quantity=prediction_quantity,
            **kwargs,
        )
        self.n_inducing_points = n_inducing_points

    def build_model(self, x_data: np.array, y_data: np.array):
        if isinstance(self.kernel, InputInitializedKernelInterface):
            self.kernel.initialize_parameters(x_data, y_data)
            self.kernel_initial_parameter_cache.clear()
            self.kernel_initial_parameter_cache.store_parameters_from_model(self.kernel)

        n_data = x_data.shape[0]
        if n_data > self.n_inducing_points:
            inducing_locations = k_means(self.n_inducing_points, x_data)
        else:
            inducing_locations = x_data.copy()
        if self.use_mean_function:
            self.model = gpflow.models.SGPR(
                data=(x_data, y_data),
                kernel=self.kernel,
                inducing_variable=inducing_locations,
                mean_function=self.mean_function,
                noise_variance=np.power(self.observation_noise, 2.0),
            )
            set_trainable(self.model.mean_function.c, False)
        else:
            self.model = gpflow.models.SGPR(
                data=(x_data, y_data),
                kernel=self.kernel,
                inducing_variable=inducing_locations,
                mean_function=None,
                noise_variance=np.power(self.observation_noise, 2.0),
            )
        set_trainable(self.model.inducing_variable, False)
        set_trainable(self.model.likelihood.variance, self.train_likelihood_variance)
        if self.set_prior_on_observation_noise:
            self.model.likelihood.variance.prior = tfd.Exponential(1 / np.power(self.expected_observation_noise, 2.0))
