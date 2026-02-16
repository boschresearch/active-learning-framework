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

from typing import List, Tuple
from alef.enums.global_model_enums import InitialParameters
from alef.models.gp_model import GPModel, PredictionQuantity
import numpy as np
from gpflow.utilities import set_trainable
from tensorflow_probability import distributions as tfd
from alef.kernels.base_object_kernel import BaseObjectKernel
from alef.models.object_gpr import ObjectGPR
from alef.utils.utils import twod_array_to_list_over_arrays
from gpflow.mean_functions import MeanFunction


class ObjectGpModel(GPModel):
    def __init__(
        self,
        kernel: BaseObjectKernel,
        observation_noise: float,
        optimize_hps: bool,
        train_likelihood_variance: bool,
        sample_initial_parameters_at_start: bool,
        initial_parameter_strategy: InitialParameters,
        perturbation_for_multistart_opt: float,
        perturbation_for_singlestart_opt: float,
        perform_multi_start_optimization: bool,
        initial_uniform_lower_bound: float,
        initial_uniform_upper_bound: float,
        n_starts_for_multistart_opt: int,
        set_prior_on_observation_noise: bool,
        expected_observation_noise: float,
        prediction_quantity: PredictionQuantity,
        **kwargs,
    ):
        super().__init__(
            kernel=kernel,
            observation_noise=observation_noise,
            optimize_hps=optimize_hps,
            train_likelihood_variance=train_likelihood_variance,
            sample_initial_parameters_at_start=sample_initial_parameters_at_start,
            initial_parameter_strategy=initial_parameter_strategy,
            perturbation_for_multistart_opt=perturbation_for_multistart_opt,
            perturbation_for_singlestart_opt=perturbation_for_singlestart_opt,
            perform_multi_start_optimization=perform_multi_start_optimization,
            initial_uniform_lower_bound=initial_uniform_lower_bound,
            initial_uniform_upper_bound=initial_uniform_upper_bound,
            n_starts_for_multistart_opt=n_starts_for_multistart_opt,
            set_prior_on_observation_noise=set_prior_on_observation_noise,
            expected_observation_noise=expected_observation_noise,
            prediction_quantity=prediction_quantity,
            **kwargs,
        )

    def build_model(self, x_data: List[object], y_data: np.array):
        if self.use_mean_function:
            self.model = ObjectGPR(
                data=(x_data, y_data),
                kernel=self.kernel,
                noise_variance=np.power(self.observation_noise, 2.0),
                mean_function=self.mean_function,
            )
        else:
            self.model = ObjectGPR(
                data=(x_data, y_data), kernel=self.kernel, noise_variance=np.power(self.observation_noise, 2.0)
            )
        set_trainable(self.model.likelihood.variance, self.train_likelihood_variance)
        if self.set_prior_on_observation_noise:
            self.model.likelihood.variance.prior = tfd.Exponential(1 / np.power(self.expected_observation_noise, 2.0))

    def set_mean_function(self, mean_function: MeanFunction):
        self.use_mean_function = True
        self.mean_function = mean_function

    def infer(self, x_data: List[object], y_data: np.array):
        x_data = self.transform_input(x_data)
        super().infer(x_data, y_data)

    def predictive_dist(self, x_test: List[object]) -> Tuple[np.array, np.array]:
        x_test = self.transform_input(x_test)
        return super().predictive_dist(x_test)

    def predictive_log_likelihood(self, x_test: List[object], y_test: np.array) -> np.array:
        x_test = self.transform_input(x_test)
        return super().predictive_log_likelihood(x_test, y_test)

    def entropy_predictive_dist(self, x_test: List[object]) -> np.array:
        x_test = self.transform_input(x_test)
        return super().entropy_predictive_dist(x_test)

    def transform_input(self, input):
        if isinstance(input, np.ndarray):
            input = twod_array_to_list_over_arrays(input)
        else:
            input = self.kernel.transform_X(input)
        return input
