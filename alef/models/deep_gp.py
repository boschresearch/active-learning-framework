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
from typing import Optional, Tuple
import numpy as np
import tensorflow as tf
from alef.models.base_model import BaseModel
from gpflux.architectures import Config, build_constant_input_dim_deep_gp
from scipy.stats import norm
from alef.utils.utils import normal_entropy

tf.keras.backend.set_floatx("float64")
tf.get_logger().setLevel("WARNING")
logger = logging.getLogger(__name__)


class DeepGP(BaseModel):
    def __init__(
        self,
        n_layer: int,
        max_n_inducing_points: int,
        learning_rate: float,
        n_iter: int,
        initial_likelihood_noise_variance: float,
        **kwargs,
    ):
        self.n_layer = n_layer
        self.max_n_inducing_points = max_n_inducing_points
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.initial_likelihood_noise_variance = initial_likelihood_noise_variance
        self.q_sqrt_scaling = 1e-5

    def build_model(self, x_data, y_data):
        if x_data.shape[0] <= self.max_n_inducing_points:
            n_inducing_points = x_data.shape[0]
        else:
            n_inducing_points = self.max_n_inducing_points
        config = Config(
            num_inducing=n_inducing_points,
            inner_layer_qsqrt_factor=self.q_sqrt_scaling,
            likelihood_noise_variance=self.initial_likelihood_noise_variance,
            whiten=True,
        )
        self.dgp = build_constant_input_dim_deep_gp(x_data, num_layers=self.n_layer, config=config)

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        logger.info("-Predict")
        prediction_model = self.dgp.as_prediction_model()
        out = prediction_model(x_test)
        mu_y = out.y_mean.numpy().squeeze()
        var_y = out.y_var.numpy().squeeze()
        sigma_y = np.sqrt(var_y)
        return mu_y, sigma_y

    def infer(self, x_data: np.array, y_data: np.array):
        logger.info("-Start training DeepGP")
        self.build_model(x_data, y_data)
        self.training_model = self.dgp.as_training_model()
        self.training_model.compile(tf.optimizers.Adam(self.learning_rate))
        self.training_model.fit({"inputs": x_data, "targets": y_data}, epochs=int(self.n_iter), verbose=0)
        logger.info("-Training finished")

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        _, pred_sigmas = self.predictive_dist(x_test)
        entropies = normal_entropy(pred_sigmas)
        return entropies

    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        pred_mus, pred_sigmas = self.predictive_dist(x_test)
        log_likelis = norm.logpdf(np.squeeze(y_test), np.squeeze(pred_mus), np.squeeze(pred_sigmas))
        return log_likelis

    def reset_model(self):
        pass

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> np.float:
        raise NotImplementedError
