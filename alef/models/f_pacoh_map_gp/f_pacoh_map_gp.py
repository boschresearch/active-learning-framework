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

import torch
import gpytorch
import time
import math
import inspect
import numpy as np
from absl import logging
from typing import List, Tuple, Dict, Optional, Callable, Union
from torch.distributions import Uniform, MultivariateNormal, kl_divergence

from alef.models.f_pacoh_map_gp.models import (
    LearnedGPRegressionModel,
    NeuralNetwork,
    AffineTransformedDistribution,
    PredictedScaleKernel,
)

from alef.models.f_pacoh_map_gp.util import _handle_input_dimensionality, DummyLRScheduler
from alef.models.f_pacoh_map_gp.abstract import RegressionModelMetaLearned
from alef.kernels.multi_output_kernels.latent_kernel_enum import LatentKernel

"""
This is from the attached code of paper (with only minor change)
Jonas Rothfuss, Christopher Koenig, Alisa Rupenyan, Andreas Krause, CoRL 2022,
Meta-Learning Priors for Safe Bayesian Optimization

The link is published by the authors
https://tinyurl.com/safe-meta-bo

see folder ./meta_bo/models/f_pacho_map.py

Copyright (c) 2022 Jonas Rothfuss, licensed under the MIT License

"""


class KernelInterface:
    def __init__(self, model, detach=False):
        self.__model = model
        self.__dumpy_points = None
        self.__kernel_std = None

    @property
    def prior_scale(self):
        if self.__kernel_std is None:
            self.__dumpy_points = self.__model.domain_dist.sample((10**4,)).numpy()
            self.__kernel_std = np.mean(
                self.__model.predict(self.__dumpy_points, return_density=False, include_obs_noise=False)[1]
            )
        return self.__kernel_std


class FPACOH_MAP_GP(RegressionModelMetaLearned):
    def __init__(
        self,
        input_dim: int,
        input_l: Union[float, np.ndarray],
        input_u: Union[float, np.ndarray],
        prior_factor: float = 1.0,
        feature_dim: int = 2,
        num_iter_fit: int = 10000,
        covar_module: str = "NN",
        mean_module: str = "NN",
        learning_mode: str = "both",
        predict_outputscale: bool = True,
        mean_nn_layers: List[int] = (32, 32, 32),
        kernel_nn_layers: List[int] = (32, 32, 32),
        prior_lengthscale: float = 0.2,
        prior_outputscale: float = 2.0,
        prior_kernel_noise: float = 1e-3,
        kernel_type: LatentKernel = LatentKernel.MATERN52,
        likelihood_std: Optional[float] = None,
        train_data_in_kl: bool = True,
        num_samples_kl: int = 20,
        task_batch_size: int = 5,
        lr: float = 1e-3,
        lr_decay: float = 1.0,
        weight_decay: float = 0.0,
        normalize_data: bool = True,
        normalization_stats: Optional[Dict] = None,
        random_state: Optional[np.random.RandomState] = None,
    ):
        super().__init__(normalize_data, random_state)
        # save init args for serialization
        self._init_args = {k: v for k, v in locals().items() if k in inspect.signature(self.__init__).parameters.keys()}

        assert learning_mode in ["learn_mean", "learn_kernel", "both", "vanilla"]
        assert mean_module in ["NN", "constant", "zero"] or isinstance(mean_module, gpytorch.means.Mean)
        assert covar_module in ["NN", "SE"] or isinstance(covar_module, gpytorch.kernels.Kernel)
        assert not predict_outputscale or covar_module == "NN", "predict_outputscale is only compatible with NN kernel"

        self.input_dim = input_dim
        self.lr, self.weight_decay, self.feature_dim = lr, weight_decay, feature_dim
        self.predict_outputscale = predict_outputscale
        self.num_iter_fit, self.task_batch_size, self.normalize_data = num_iter_fit, task_batch_size, normalize_data

        """ Setup prior and likelihood """
        self._setup_gp_prior(
            mean_module, covar_module, learning_mode, feature_dim, mean_nn_layers, kernel_nn_layers, kernel_type
        )
        self.kernel = KernelInterface(self, not predict_outputscale)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.likelihoods.noise_models.GreaterThan(1e-4)
        )
        if likelihood_std is None:
            self.likelihood.noise = 0.1**2
            self.shared_parameters.append({"params": self.likelihood.parameters(), "lr": self.lr})
        else:
            self.likelihood.noise = likelihood_std**2
        self._setup_optimizer(lr, lr_decay)

        """ domain support dist & prior kernel """
        self.prior_factor = prior_factor
        bound_l = self.__check_domain_bound(input_l)
        bound_u = self.__check_domain_bound(input_u)
        self.domain_dist = Uniform(low=torch.from_numpy(bound_l).float(), high=torch.from_numpy(bound_u).float())
        self.train_data_in_kl = train_data_in_kl
        self.num_samples_kl = num_samples_kl

        base_kernel = self._get_base_kernel(kernel_type, input_dims=self.input_dim)
        self.prior_covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        self.prior_covar_module.outputscale = prior_outputscale
        self.prior_covar_module.base_kernel.lengthscale = prior_lengthscale
        self.prior_kernel_noise = prior_kernel_noise

        """ ------- normalization stats & data setup  ------- """
        self._normalization_stats = normalization_stats
        self.reset_to_prior()

        self.fitted = False

    def __check_domain_bound(self, bound: Union[float, np.ndarray]):
        if hasattr(bound, "__len__"):
            assert len(bound) == self.input_dim, f"bound {bound} is not compatible with input_dim {self.input_dim}"
            return bound
        else:
            return np.array([bound] * self.input_dim)

    def meta_fit(
        self,
        meta_train_tuples: List[Tuple[np.ndarray, np.ndarray]],
        meta_valid_tuples: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None,
        verbose: bool = True,
        log_period: int = 500,
        n_iter: Optional[int] = None,
    ):
        assert (meta_valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in meta_valid_tuples]))
        self.likelihood.train()

        task_dicts = self._prepare_meta_train_tasks(meta_train_tuples)

        t = time.time()
        cum_loss = 0.0
        n_iter = self.num_iter_fit if n_iter is None else n_iter

        for itr in range(1, n_iter + 1):
            # actual meta-training step
            task_dict_batch = self._rds.choice(task_dicts, size=self.task_batch_size)
            loss = self._step(task_dict_batch, n_tasks=len(task_dicts))
            cum_loss += loss

            # print training stats stats
            if itr == 1 or itr % log_period == 0:
                duration = time.time() - t
                avg_loss = cum_loss / (log_period if itr > 1 else 1.0)
                cum_loss = 0.0
                t = time.time()

                message = "Iter %d/%d - Loss: %.6f - Time %.2f sec" % (itr, self.num_iter_fit, avg_loss, duration)

                # if validation data is provided  -> compute the valid log-likelihood
                if meta_valid_tuples is not None:
                    self.likelihood.eval()
                    valid_ll, valid_rmse, calibr_err, calibr_err_chi2 = self.eval_datasets(meta_valid_tuples)
                    self.likelihood.train()
                    message += " - Valid-LL: %.3f - Valid-RMSE: %.3f - Calib-Err %.3f" % (
                        valid_ll,
                        valid_rmse,
                        calibr_err,
                    )
                if verbose:
                    logging.info(message)

        self.fitted = True

        # set gpytorch modules to eval mode and set gp to meta-learned gp prior
        assert self.X_data.shape[0] == 0 and self.y_data.shape[0] == 0, (
            "Data for posterior inference can be passed only after the meta-training"
        )
        for task_dict in task_dicts:
            task_dict["model"].eval()
        self.likelihood.eval()
        self.reset_to_prior()
        return loss

    def predict(self, test_x: np.ndarray, return_density: bool = False, include_obs_noise: bool = True, **kwargs):
        test_x = _handle_input_dimensionality(test_x)

        with torch.no_grad():
            test_x_normalized = self._normalize_data(test_x)
            test_x_tensor = torch.from_numpy(test_x_normalized).float()

            pred_dist = self.gp(test_x_tensor)
            if include_obs_noise:
                pred_dist = self.likelihood(pred_dist)
            pred_dist_transformed = AffineTransformedDistribution(
                pred_dist, normalization_mean=self.y_mean, normalization_std=self.y_std
            )
            if return_density:
                return pred_dist_transformed
            else:
                pred_mean = pred_dist_transformed.mean.cpu().numpy()
                pred_std = pred_dist_transformed.stddev.cpu().numpy()
                return pred_mean, pred_std

    def predict_mean_std(self, test_x):
        return self.predict(test_x, return_density=False)

    def meta_predict(self, context_x, context_y, test_x, return_density=False):
        """
        Performs posterior inference (target training) with (context_x, context_y) as training data and then
        computes the predictive distribution of the targets p(y|test_x, test_context_x, context_y) in the test points

        Args:
            context_x: (ndarray) context input data for which to compute the posterior
            context_y: (ndarray) context targets for which to compute the posterior
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density: (bool) whether to return result as mean and std ndarray or as MultivariateNormal pytorch object

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(t|test_x, test_context_x, context_y)
        """

        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        test_x = _handle_input_dimensionality(test_x)
        assert test_x.shape[1] == context_x.shape[1]

        # normalize data and convert to tensor
        context_x, context_y = self._prepare_data_per_task(context_x, context_y)

        test_x = self._normalize_data(X=test_x, Y=None)
        test_x = torch.from_numpy(test_x).float()

        with torch.no_grad():
            # compute posterior given the context data
            gp_model = LearnedGPRegressionModel(
                context_x,
                context_y,
                self.likelihood,
                learned_kernel=self.nn_kernel_map,
                learned_mean=self.nn_mean_fn,
                covar_module=self.covar_module,
                mean_module=self.mean_module,
            )
            gp_model.eval()
            self.likelihood.eval()
            pred_dist = self.likelihood(gp_model(test_x))
            pred_dist_transformed = AffineTransformedDistribution(
                pred_dist, normalization_mean=self.y_mean, normalization_std=self.y_std
            )

        if return_density:
            return pred_dist_transformed
        else:
            pred_mean = pred_dist_transformed.mean
            pred_std = pred_dist_transformed.stddev
            return pred_mean.cpu().numpy(), pred_std.cpu().numpy()

    def pred_mean_grad(self, test_x: np.ndarray) -> np.ndarray:
        test_x = _handle_input_dimensionality(test_x)
        test_x_normalized = self._normalize_data(test_x)
        test_x_tensor = torch.tensor(test_x_normalized).float()
        test_x_tensor.requires_grad = True
        pred_dist = self.gp(test_x_tensor)
        pred_dist_transformed = AffineTransformedDistribution(
            pred_dist, normalization_mean=self.y_mean, normalization_std=self.y_std
        )
        pred_mean = pred_dist_transformed.mean.sum()
        pred_mean.backward()
        pred_mean_grad_x = test_x_tensor.grad.detach().numpy() * self.x_std[None, :]
        assert pred_mean_grad_x.shape == test_x_normalized.shape
        return pred_mean_grad_x

    @property
    def gp(self) -> Callable:
        if self.X_data.shape[0] > 0:
            return self._gp
        else:
            return self._prior

    def reset_to_prior(self):
        self._reset_data()
        self._gp = None

    @property
    def likelihood_std(self) -> float:
        return float(self.likelihood.noise**0.5)

    def prior(self, x: np.ndarray, return_density: bool = False):
        x = _handle_input_dimensionality(x)
        with torch.no_grad():
            x_normalized = self._normalize_data(x)
            x_tensor = torch.from_numpy(x_normalized).float()
            prior_dist = self._prior(x_tensor)
            prior_dist_transformed = AffineTransformedDistribution(
                prior_dist, normalization_mean=self.y_mean, normalization_std=self.y_std
            )
            if return_density:
                return prior_dist_transformed
            else:
                return prior_dist_transformed.mean.cpu().numpy(), prior_dist_transformed.stddev.cpu().numpy()

    def _reset_posterior(self):
        x_context = torch.from_numpy(self.X_data).float()
        y_context = torch.from_numpy(self.y_data).float()
        self._gp = LearnedGPRegressionModel(
            x_context,
            y_context,
            self.likelihood,
            learned_kernel=self.nn_kernel_map,
            learned_mean=self.nn_mean_fn,
            covar_module=self.covar_module,
            mean_module=self.mean_module,
        )
        self._gp.eval()
        self.likelihood.eval()

    def _prior(self, x: torch.Tensor) -> torch.distributions.Distribution:
        if self.nn_kernel_map is not None:
            projected_x = self.nn_kernel_map(x)
        else:
            projected_x = x

            # feed through mean module
        if self.nn_mean_fn is not None:
            mean_x = self.nn_mean_fn(x).squeeze()
        else:
            mean_x = self.mean_module(projected_x).squeeze()

        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _prepare_meta_train_tasks(self, meta_train_tuples):
        self._check_meta_data_shapes(meta_train_tuples)

        if self._normalization_stats is None:
            self._compute_meta_normalization_stats(meta_train_tuples)
        else:
            self._set_normalization_stats(self._normalization_stats)

        self.x_mean_torch = torch.from_numpy(self.x_mean).float()
        self.x_std_torch = torch.from_numpy(self.x_std).float()
        task_dicts = [self._dataset_to_task_dict(x, y) for x, y in meta_train_tuples]
        return task_dicts

    def _dataset_to_task_dict(self, x, y):
        # a) prepare data
        x_tensor, y_tensor = self._prepare_data_per_task(x, y)
        task_dict = {"x_train": x_tensor, "y_train": y_tensor}

        # b) prepare model
        task_dict["model"] = LearnedGPRegressionModel(
            task_dict["x_train"],
            task_dict["y_train"],
            self.likelihood,
            learned_kernel=self.nn_kernel_map,
            learned_mean=self.nn_mean_fn,
            covar_module=self.covar_module,
            mean_module=self.mean_module,
        )
        task_dict["mll_fn"] = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, task_dict["model"])
        return task_dict

    def _step(self, task_dict_batch, n_tasks):
        assert len(task_dict_batch) > 0
        loss = 0.0
        self.optimizer.zero_grad()

        for task_dict in task_dict_batch:
            # mll term
            output = task_dict["model"](task_dict["x_train"])
            mll = task_dict["mll_fn"](output, task_dict["y_train"])

            # kl term
            kl = self._f_kl(task_dict)

            #  terms for pre-factors
            n = n_tasks
            m = task_dict["x_train"].shape[0]

            # loss for this batch
            loss += (
                -mll / self.task_batch_size
                + self.prior_factor * (1 / math.sqrt(n) + 1 / (n * m)) * kl / self.task_batch_size
            )

        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss.item()

    def _sample_measurement_set(self, x_train):
        if self.train_data_in_kl:
            n_train_x = min(x_train.shape[0], self.num_samples_kl // 2)
            n_rand_x = self.num_samples_kl - n_train_x
            idx_rand = np.random.choice(x_train.shape[0], n_train_x)
            x_domain_dist = self._normalize_x_torch(self.domain_dist.sample((n_rand_x,)))
            x_kl = torch.cat([x_train[idx_rand], x_domain_dist], dim=0)
        else:
            x_kl = self._normalize_x_torch(self.domain_dist.sample((self.num_samples_kl,)))
        assert x_kl.shape == (self.num_samples_kl, self.input_dim)
        return x_kl

    def _normalize_x_torch(self, X):
        assert hasattr(self, "x_mean_torch") and hasattr(self, "x_std_torch"), (
            "requires computing normalization stats beforehand"
        )
        x_normalized = (X - self.x_mean_torch[None, :]) / self.x_std_torch[None, :]
        return x_normalized

    def _f_kl(self, task_dict):
        with gpytorch.settings.debug(False):
            # sample / construc measurement set
            x_kl = self._sample_measurement_set(task_dict["x_train"])

            # functional KL
            dist_f_posterior = task_dict["model"](x_kl)
            K_prior = torch.reshape(self.prior_covar_module(x_kl).evaluate(), (x_kl.shape[0], x_kl.shape[0]))

            inject_noise_std = self.prior_kernel_noise
            error_counter = 0
            while error_counter < 5:
                try:
                    dist_f_prior = MultivariateNormal(
                        torch.zeros(x_kl.shape[0]),
                        scale_tril=torch.linalg.cholesky(K_prior + inject_noise_std * torch.eye(x_kl.shape[0])),
                    )
                    return kl_divergence(dist_f_posterior, dist_f_prior)
                except RuntimeError as e:
                    import warnings

                    inject_noise_std = 2 * inject_noise_std
                    error_counter += 1
                    warnings.warn(
                        "encoundered numerical error in computation of KL: %s "
                        "--- Doubling inject_noise_std to %.4f and trying again" % (str(e), inject_noise_std)
                    )
            raise RuntimeError("Not able to compute KL")

    def _setup_gp_prior(
        self, mean_module, covar_module, learning_mode, feature_dim, mean_nn_layers, kernel_nn_layers, kernel_type
    ):
        self.shared_parameters = []

        # a) determine kernel map & module
        if covar_module == "NN":
            assert learning_mode in ["learn_kernel", "both"], "neural network parameters must be learned"
            # If necessary, add one dimension that encodes the predicted output scale to the feature space
            feature_dim_nn = feature_dim + 1 if self.predict_outputscale else feature_dim
            self.nn_kernel_map = NeuralNetwork(
                input_dim=self.input_dim, output_dim=feature_dim_nn, layer_sizes=kernel_nn_layers
            )
            self.shared_parameters.append(
                {"params": self.nn_kernel_map.parameters(), "lr": self.lr, "weight_decay": self.weight_decay}
            )
            base_kernel = self._get_base_kernel(kernel_type, feature_dim)
            if self.predict_outputscale:
                self.covar_module = PredictedScaleKernel(base_kernel)
            else:
                self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        else:
            self.nn_kernel_map = None

        if covar_module == "SE":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.input_dim))
        elif isinstance(covar_module, gpytorch.kernels.Kernel):
            self.covar_module = covar_module

        # b) determine mean map & module

        if mean_module == "NN":
            assert learning_mode in ["learn_mean", "both"], "neural network parameters must be learned"
            self.nn_mean_fn = NeuralNetwork(input_dim=self.input_dim, output_dim=1, layer_sizes=mean_nn_layers)
            self.shared_parameters.append(
                {"params": self.nn_mean_fn.parameters(), "lr": self.lr, "weight_decay": self.weight_decay}
            )
            self.mean_module = None
        else:
            self.nn_mean_fn = None

        if mean_module == "constant":
            self.mean_module = gpytorch.means.ConstantMean()
        elif mean_module == "zero":
            self.mean_module = gpytorch.means.ZeroMean()
        elif isinstance(mean_module, gpytorch.means.Mean):
            self.mean_module = mean_module

        # c) add parameters of covar and mean module if desired

        if learning_mode in ["learn_kernel", "both"]:
            self.shared_parameters.append({"params": self.covar_module.hyperparameters(), "lr": self.lr})

        if learning_mode in ["learn_mean", "both"] and self.mean_module is not None:
            self.shared_parameters.append({"params": self.mean_module.hyperparameters(), "lr": self.lr})

    def _setup_optimizer(self, lr, lr_decay):
        self.optimizer = torch.optim.AdamW(self.shared_parameters, lr=lr, weight_decay=self.weight_decay)
        if lr_decay < 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, gamma=lr_decay)
        else:
            self.lr_scheduler = DummyLRScheduler()

    def _vectorize_pred_dist(self, pred_dist):
        return torch.distributions.Normal(pred_dist.mean, pred_dist.stddev)

    def _get_base_kernel(self, kernel_type: LatentKernel, input_dims: int) -> gpytorch.kernels.Kernel:
        if kernel_type == LatentKernel.RBF:
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dims)
        elif kernel_type == LatentKernel.MATERN52:
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dims)
        elif kernel_type == LatentKernel.MATERN32:
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=input_dims)
        elif kernel_type == LatentKernel.MATERN12:
            base_kernel = gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=input_dims)
        else:
            raise ValueError(f"Unknown kernel type {kernel_type}")
        return base_kernel

    @classmethod
    def select_hparam_via_bracket_cv(
        cls,
        default_kwargs: Dict,
        meta_train_data: List[Tuple[np.ndarray, np.ndarray]],
        target_param: str = "prior_factor",
        min_calib_freq: float = 0.995,
        logspace: bool = True,
        upper: float = 10.0,
        lower: float = 1e-1,
        num_iters: int = 6,
        verbose: bool = True,
        increase_when_uncalibrated=True,
    ) -> object:
        return super().select_hparam_via_bracket_cv(
            default_kwargs,
            meta_train_data,
            target_param=target_param,
            min_calib_freq=min_calib_freq,
            logspace=logspace,
            upper=upper,
            lower=lower,
            num_iters=num_iters,
            verbose=verbose,
            increase_when_uncalibrated=increase_when_uncalibrated,
        )


if __name__ == "__main__":
    from alef.oracles import BraninHoo, GPOracle1D
    from alef.kernels.kernel_factory import KernelFactory
    from alef.configs.kernels.matern52_configs import BasicMatern52Config

    x = torch.tensor(np.array([[0.0]]))
    print(x.size)
    oracle_s1 = BraninHoo(
        0.01,
        constants=np.array(
            [
                1.429616092817147965e00,
                1.158187777290893006e-01,
                1.183918811677094451e00,
                5.409120557106079197e00,
                1.027090011632674660e01,
                4.191089405958503544e-02,
            ]
        ),
        normalize_mean=7.579077429606144278e01,
        normalize_scale=7.848612060810351920e01,
    )
    oracle_s2 = BraninHoo(
        0.01,
        constants=np.array(
            [
                1.212358667760751851e00,
                1.414457171919928091e-01,
                1.886480860828303685e00,
                5.440170505117864153e00,
                1.081822960287230906e01,
                4.659429846709679801e-02,
            ]
        ),
        normalize_mean=8.118478514218084285e01,
        normalize_scale=8.773486869249032338e01,
    )
    oracle_s3 = BraninHoo(
        0.01,
        constants=np.array(
            [
                7.236089185286261882e-01,
                1.144487842459822602e-01,
                1.427798409117466250e00,
                6.123925824119192285e00,
                1.075253092811846400e01,
                3.063985069621295854e-02,
            ]
        ),
        normalize_mean=4.579533942550646231e01,
        normalize_scale=4.134298150208427813e01,
    )
    oracle_s4 = BraninHoo(
        0.01,
        constants=np.array(
            [
                1.080619357765209676e00,
                1.299557365724782754e-01,
                1.425199351563181782e00,
                5.848237025003893308e00,
                1.181030676579931260e01,
                3.457920592528983261e-02,
            ]
        ),
        normalize_mean=6.645792281947375102e01,
        normalize_scale=6.657196332123434956e01,
    )
    oracle_s5 = BraninHoo(
        0.01,
        constants=np.array(
            [
                1.447419006839536504e00,
                1.402526533934428665e-01,
                1.587797568950959359e00,
                6.188635171723014139e00,
                8.928569161987805813e00,
                4.285981453000599350e-02,
            ]
        ),
        normalize_mean=8.314143012705224578e01,
        normalize_scale=9.500427063005324158e01,
    )

    oracle = BraninHoo(0.01)

    oracle = GPOracle1D(
        KernelFactory.build(BasicMatern52Config(input_dimension=1, base_lengthscale=[0.1], base_variance=1)),
        noise_level=0.1,
    )
    l = 0
    u = 1
    meta_train_data = []
    for i in range(50):
        oracle.initialize(l, u, 200)
        meta_train_data.append(oracle.get_random_data(30, noisy=True))

    meta_train_data_calib = []
    for i in range(20):
        oracle.initialize(l, u, 200)
        meta_train_data_calib.append(oracle.get_random_data(50, noisy=True))

    meta_test_data = []
    for i in range(50):
        oracle.initialize(l, u, 200)
        meta_test_data.append((*oracle.get_random_data(10, noisy=True), *oracle.get_random_data(160, noisy=True)))

    NN_LAYERS = [32, 32]

    plot = False
    from matplotlib import pyplot as plt

    if plot:
        for x_train, y_train in meta_train_data:
            plt.scatter(x_train, y_train)
        plt.title("sample from the GP prior")
        plt.show()

    """ 2) Classical mean learning based on mll """

    print("\n ---- GPR mll meta-learning ---- ")

    torch.set_num_threads(2)

    prior_factor = 0.1
    gp_model = FPACOH_MAP_GP(
        oracle.get_dimension(),
        l,
        u,
        num_iter_fit=4000,
        weight_decay=5e-4,
        prior_factor=prior_factor,
        task_batch_size=2,
        covar_module="NN",
        mean_module="NN",
        prior_outputscale=9.0,
        predict_outputscale=True,
        kernel_type=LatentKernel.MATERN52,
        prior_lengthscale=0.15,
        mean_nn_layers=NN_LAYERS,
        kernel_nn_layers=NN_LAYERS,
        normalization_stats=None,
    )
    itrs = 0
    for i in range(3):
        gp_model.meta_fit(meta_train_data, meta_valid_tuples=None, log_period=1000, n_iter=500)
        itrs += 400

        calib_freq, avg_std = gp_model.meta_calibration_sharpness_for_datasets(
            meta_train_data_calib, min_conf_level=0.8
        )

        print(
            f"iter {itrs}: | likelihood_std = {gp_model.likelihood_std} | calibr_freq = {calib_freq} | avg_std = {avg_std}"
        )

        """ plotting """
        fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
        x_plot = np.linspace(l, u, num=150)

        # prior
        prior_mean, priod_std = gp_model.prior(x_plot)
        axes[0].plot(x_plot, prior_mean)
        axes[0].fill_between(x_plot.flatten(), prior_mean - 2 * priod_std, prior_mean + 2 * priod_std, alpha=0.2)
        axes[0].set_title("prior")

        # posterior
        x_context, t_context, x_test, y_test = meta_test_data[0]
        pred_mean, pred_std = gp_model.meta_predict(x_context, t_context, x_plot)
        ucb, lcb = (pred_mean + 2 * pred_std).flatten(), (pred_mean - 2 * pred_std).flatten()

        axes[1].scatter(x_test, y_test)
        axes[1].scatter(x_context, t_context)
        axes[1].plot(x_plot, pred_mean)
        axes[1].fill_between(x_plot.flatten(), lcb, ucb, alpha=0.2)
        axes[1].set_title("posterior")

        fig.suptitle(f"GPR meta mll (prior_factor =  {prior_factor}) itrs = {itrs}")
