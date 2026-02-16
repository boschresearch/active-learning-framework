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

import numpy as np
import tensorflow as tf
from gpflow.kernels import MultioutputKernel
from gpflow.inducing_variables import (
    SharedIndependentInducingVariables,
    SeparateIndependentInducingVariables,
    InducingPoints,
)
from typing import Tuple, Optional
from alef.models.base_model import BaseModel
from gpflow.models import SVGP
import gpflow
import time
from alef.enums.global_model_enums import PredictionQuantity


def iv_array_init(input_dim, M):
    Z_init = np.tile(np.linspace(-1, 1, M)[:, None], (1, input_dim))
    return Z_init


class SVGPModel(BaseModel, SVGP):
    def __init__(
        self,
        kernel,
        likelihood,
        M: int,
        input_dim: int,
        *,
        share_inducing_points: bool = True,
        mean_function=None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
        whiten: bool = True,
        num_data: int = None,
        prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y,
    ):
        if isinstance(kernel, MultioutputKernel):
            num_latent_gps = kernel.num_latent_gps

        if share_inducing_points:
            Z = iv_array_init(input_dim, M)
            iv = SharedIndependentInducingVariables(InducingPoints(Z))
        else:
            Zs = [iv_array_init(input_dim, M) for _ in range(num_latent_gps)]
            iv_list = [InducingPoints(Z) for Z in Zs]
            iv = SeparateIndependentInducingVariables(iv_list)

        super().__init__(
            kernel,
            likelihood,
            iv,
            mean_function=mean_function,
            num_latent_gps=num_latent_gps,
            q_diag=q_diag,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            whiten=whiten,
            num_data=num_data,
        )

        self.prediction_quantity = prediction_quantity

    def reset_model(self):
        pass

    def set_optimizer(self, opt: str = "adam", **kwargs):
        self.opt = opt
        self.opt_args = kwargs

    def infer(self, x_data: np.array, y_data: np.array):
        """
        Performs inference of the model - trains all parameters and latent variables needed for prediction

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        y_data: Label array with shape (n,1) where n is the number of training points
        """
        loss = self.training_loss_closure((x_data, y_data))

        opt_args = self.opt_args

        if self.opt.lower() == "scipy":
            timer = -time.perf_counter()
            gpflow.optimizers.Scipy().minimize(loss, variables=self.trainable_variables, **opt_args)
            timer += time.perf_counter()

        elif self.opt.lower() == "adam":
            MAXITER = 500
            optimizer = tf.optimizers.Adam(learning_rate=0.1)

            timer = -time.perf_counter()
            for _ in range(MAXITER):
                optimizer.minimize(loss, self.trainable_variables)
            timer += time.perf_counter()

        else:
            raise ValueError("unknown optimizer")

        return timer

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        """
        Method for retrieving the predictive mean and sigma for a given array of the test points

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        mean array with shape (n,p)
        sigma array with shape (n,p)
        """
        if self.prediction_quantity == PredictionQuantity.PREDICT_F:
            pred_mus, pred_vars = self.predict_f(x_test)
        elif self.prediction_quantity == PredictionQuantity.PREDICT_Y:
            pred_mus, pred_vars = self.predict_y(x_test)
        pred_sigmas = np.sqrt(pred_vars)
        return np.squeeze(pred_mus), np.squeeze(pred_sigmas)

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> np.float:
        """
        Method for estimating the model evidence (as we only use bayesian models this should in principle be possible)

        Arguments:
        x_data: Optional - Input array with shape (n,d) where d is the input dimension and n the number of training points
        y_data: Optional - Label array with shape (n,1) where n is the number of training points

        Returns:
        evidence value - single value
        """

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        """
        Method for calculating the entropy of the predictive distribution for test sequence - used for acquistion function in active learning

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        entropy array with shape (n,1)
        """
        _, cov = self.predict_f(x_test, full_output_cov=True)
        cov = cov.numpy()

        entropy = 0.5 * np.log((2 * np.pi * np.e) ** cov.shape[-1] * np.linalg.det(cov))
        entropy = entropy.reshape([-1, 1])

        return entropy
