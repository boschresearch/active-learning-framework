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
import time
from typing import Tuple, Optional

from alef.models.base_model import BaseModel
from alef.kernels.multi_output_kernels.base_transfer_kernel import BaseTransferKernel
from alef.enums.global_model_enums import PredictionQuantity
from alef.models.f_pacoh_map_gp.prior_gp import GPRegressionVanilla
from alef.models.f_pacoh_map_gp.f_pacoh_map_gp import FPACOH_MAP_GP
import logging

logger = logging.getLogger(__name__)

"""
We implement the metaGP described in
Jonas Rothfuss, Christopher Koenig, Alisa Rupenyan, Andreas Krause, CoRL 2022,
Meta-Learning Priors for Safe Bayesian Optimization

This model wrap their code (see ./f_pacoh_map_gp/*) into our interface
"""


class MetaGPModel(BaseModel):
    """

    Attributes:
        kernel:
    """

    def __init__(
        self,
        zero_mean: bool,
        kernel: BaseTransferKernel,
        observation_noise: float,
        min_calib_freq: float,
        weight_decay: float,
        num_iter_fit: int,
        predict_outputscale: bool,
        train_data_in_kl: bool,
        optimize_hps: bool,
        prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_F,
        **kwargs,
    ):
        self.zero_mean = zero_mean
        self.kernel = kernel
        self.prior_gp_model = None
        self.model = None
        self.meta_trained = False
        self.data = None
        self.observation_noise = observation_noise
        self.min_calib_freq = min_calib_freq
        self.weight_decay = weight_decay
        self.num_iter_fit = num_iter_fit
        self.predict_outputscale = predict_outputscale
        self.train_data_in_kl = train_data_in_kl
        self.optimize_hps = optimize_hps
        self.prediction_quantity = prediction_quantity
        self.set_input_domain(0.0, 1.0)

    def set_input_domain(self, l, u):
        self.input_l = l
        self.input_u = u
        self.reset_model(reset_source=True)

    def reset_model(self, reset_source: bool = False):
        """
        resets the model to the initial values - kernel parameters and observation noise are reset to initial values - gpflow model is deleted
        """
        if reset_source:
            self.meta_trained = False
            self.prior_gp_model = None
            self.model = None

    def infer(self, x_data: np.array, y_data: np.array):
        """
        Main entrance method for learning the model - training methods are called if hp should be done otherwise only gpflow model is build

        Arguments:
            x_data: Input array with shape (n,d+1) where d is the input dimension and n the number of training points
            y_data: Label array with shape (n,1) where n is the number of training points
        """
        P = self.kernel.output_dimension
        mask = x_data[:, -1] == (P - 1)
        self.data = (x_data[mask], y_data[mask])
        if self.meta_trained:
            return 0
        else:
            P = self.kernel.output_dimension
            meta_data = []
            for s in range(P):
                mask = x_data[:, -1] == s
                meta_data.append((x_data[mask, :-1], y_data[mask, 0]))
            t0 = time.perf_counter()
            self.prior_gp_model = GPRegressionVanilla.select_kernel_via_frontier_cv(
                {
                    "input_dim": self.kernel.input_dimension,
                    "kernel_lengthscale": self.kernel.base_lengthscale,
                    "kernel_variance": self.kernel.base_variance,
                    "likelihood_std": self.observation_noise,
                    "normalize_data": True,
                    "normalization_stats": None,
                },
                meta_train_data=meta_data,
                min_calib_freq=self.min_calib_freq,
            )
            self.model = FPACOH_MAP_GP(
                self.kernel.input_dimension,
                self.input_l,
                self.input_u,
                mean_module="NN" if not self.zero_mean else "zero",
                weight_decay=self.weight_decay,
                num_iter_fit=self.num_iter_fit,
                predict_outputscale=self.predict_outputscale,
                kernel_type=self.kernel.kernel_type,
                likelihood_std=self.observation_noise,
                train_data_in_kl=self.train_data_in_kl,
                normalization_stats=None,
                prior_lengthscale=self.prior_gp_model.kernel_lengthscale,
                prior_outputscale=self.prior_gp_model.kernel_outputscale,
            )

            self.model.meta_fit(meta_data, meta_valid_tuples=None, log_period=500)
            t1 = time.perf_counter()
            self.meta_trained = True
            return t1 - t0

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> float:
        """
        Estimates the model evidence - always retrieves marg likelihood, also when HPs are provided with prior!!

        Arguments:
            x_data: (only necessary if infer was not yet called) Input array with shape (n,d) where d is the input dimension and n the number of training points
            y_data: (only necessary if infer was not yet called) Label array with shape (n,1) where n is the number of training points

        Returns:
            marginal likelihood value of infered model
        """
        raise NotImplementedError

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        """
        Method for retrieving the predictive mean and sigma for a given array of the test points

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        mean array with shape (n,m)
        sigma array with shape (n,m)
        """
        D = self.kernel.input_dimension
        P = self.kernel.output_dimension
        assert x_test.shape[1] == D or np.all(x_test[:, D] == (P - 1))
        context_x, context_y = self.data
        if self.prediction_quantity == PredictionQuantity.PREDICT_F:
            # pred_mus, pred_vars = self.model.predict(x_test[:, :D], return_density=False, include_obs_noise=False)
            pred_mus, pred_sigmas = self.model.meta_predict(
                context_x[:, :D], context_y[:, :D], x_test[:, :D], return_density=False
            )
        elif self.prediction_quantity == PredictionQuantity.PREDICT_Y:
            pred_mus, pred_sigmas = self.model.meta_predict(
                context_x[:, :D], context_y[:, :D], x_test[:, :D], return_density=False
            )
        return np.squeeze(pred_mus), np.squeeze(pred_sigmas)

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        """
        Method for calculating the entropy of the predictive distribution for test sequence - used for acquistion function in active learning

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        entropy array with shape (n,1)
        """
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
        raise NotImplementedError


if __name__ == "__main__":
    from alef.kernels.kernel_factory import KernelFactory
    from alef.configs.kernels.multi_output_kernels.fpacoh_kernel_config import BasicFPACOHKernelConfig
    from alef.experiments.safe_learning.simulator import PoolHandler
    from alef.enums.global_model_enums import PredictionQuantity
    import matplotlib.pyplot as plt

    pool = PoolHandler().get_transfer_pool_from_oracle(
        3000, oracle_type="illustrate", observation_noise=0.1, additional_safety=False
    )
    kernel = KernelFactory.build(
        BasicFPACOHKernelConfig(
            base_variance=1.0, base_lengthscale=1.0, input_dimension=pool.get_dimension(), output_dimension=2
        )
    )
    model = MetaGPModel(
        zero_mean=True,
        kernel=kernel,
        observation_noise=0.1,
        min_calib_freq=0.9,
        weight_decay=1e-4,
        num_iter_fit=500,
        predict_outputscale=True,
        train_data_in_kl=True,
        optimize_hps=True,
        prediction_quantity=PredictionQuantity.PREDICT_Y,
    )
    model.set_input_domain(*pool.target_pool.get_box_bounds())

    for i in range(1):
        print(f"iter {i}")
        pool.set_task_mode(learning_target=False)
        Xs, Ys = pool.get_random_constrained_data(50, constraint_lower=0.0)
        pool.set_task_mode(learning_target=True)
        Xt, Yt = pool.get_random_constrained_data_in_box(5, -1, 0.5, constraint_lower=0.0)

        X = np.vstack((Xs, Xt))
        Y = np.vstack((Ys, Yt))
        print("   train source")
        model.infer(Xs, Ys)
        print("   predict")
        mu, std = model.predictive_dist(Xt)

        # model.reset_model()
        print("   train target")
        model.infer(X, Y)
        print("   predict")
        mu, std = model.predictive_dist(Xt)

        fig, axs = plt.subplots(1, 1)
        axs.plot(Xs[:, 0], Ys[:, 0], "y.")
        axs.plot(Xt[:, 0], Yt[:, 0], "b.")
        axs.plot(Xt[:, 0], mu, "b-")
        idx = np.argsort(Xt[:, 0])
        axs.fill_between(Xt[idx, 0], mu[idx] + std[idx], mu[idx] - std[idx], color="b", alpha=0.3)
        fig.savefig("thisisatest.png")
