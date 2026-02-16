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

from typing import List, Optional, Tuple
import gpflow
import numpy as np
from alef.kernels.deep_kernels.base_deep_kernel import BaseDeepKernel
from alef.models.base_model import BaseModel
from gpflow.utilities import positive
import tensorflow as tf
import logging

from alef.models.gp_model import GPModel

logger = logging.getLogger(__name__)


class DeepKernelTransfer(BaseModel):
    """
    Very basic implementation of
    """

    def __init__(self, deep_kernel: BaseDeepKernel) -> None:
        super().__init__()
        self.deep_kernel = deep_kernel
        self.initial_observation_noise = 0.01
        self.initial_noise_variance = np.power(self.initial_observation_noise, 2.0)
        self.noise_variance_parameter = gpflow.Parameter(self.initial_noise_variance, transform=positive(lower=1e-6))
        self.learning_rate = 0.01
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

    def build_models(self, tasks_list: List[Tuple[np.array, np.array]]):
        gpr_models = []
        for x_data, y_data in tasks_list:
            model = gpflow.models.GPR(
                data=(x_data, y_data),
                kernel=self.deep_kernel,
                mean_function=None,
                noise_variance=self.initial_noise_variance,
            )
            model.likelihood.variance = self.noise_variance_parameter
            gpr_models.append(model)
        return gpr_models

    def meta_train(self, n_episodes, tasks_list: List[Tuple[np.array, np.array]]):
        gpr_model_list = self.build_models(tasks_list)
        for i in range(0, n_episodes):
            for i, gpr_model in enumerate(gpr_model_list):
                training_loss = gpr_model.training_loss_closure()
                self.optimizer.minimize(training_loss, gpr_model.trainable_variables)
                loss_value = training_loss().numpy()
                logger.debug("Loss value for task " + str(i) + ": " + str(loss_value))

    def infer(self, x_data: np.array, y_data: np.array):
        observation_noise = np.sqrt(self.noise_variance_parameter.numpy())
        self.transfered_gp_model = GPModel(self.deep_kernel, observation_noise, False, False)
        self.transfered_gp_model.infer(x_data, y_data)

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        return self.transfered_gp_model.predictive_dist(x_test)

    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        return self.transfered_gp_model.predictive_log_likelihood(x_test, y_test)

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        return self.transfered_gp_model.entropy_predictive_dist(x_test)

    def reset_model(self):
        pass

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> np.float:
        pass
