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

import gpytorch
import copy
import numpy as np
import inspect
import torch

from alef.models.f_pacoh_map_gp.models import LearnedGPRegressionModel, AffineTransformedDistribution
from alef.models.f_pacoh_map_gp.abstract import RegressionModel
from alef.models.f_pacoh_map_gp.bracketing import MonotoneFrontierSolver, MonotoneFrontierSolverV2
from alef.kernels.multi_output_kernels.latent_kernel_enum import LatentKernel
from typing import Tuple, Dict, Optional, List, Callable

"""
This is from the attached code of paper
Jonas Rothfuss, Christopher Koenig, Alisa Rupenyan, Andreas Krause, CoRL 2022,
Meta-Learning Priors for Safe Bayesian Optimization

The link is published by the authors
https://tinyurl.com/safe-meta-bo

see folder ./meta_bo/models/vanilla_gp.py

Copyright (c) 2022 Jonas Rothfuss, licensed under the MIT License

"""


class GPRegressionVanilla(RegressionModel):
    def __init__(
        self,
        input_dim: int,
        kernel_variance: float = 1.0,
        kernel_lengthscale: float = 0.2,
        likelihood_std: float = 0.1,
        normalize_data: bool = True,
        normalization_stats: Optional[Dict] = None,
        kernel_type: LatentKernel = LatentKernel.MATERN52,
        random_state: Optional[np.random.RandomState] = None,
    ):
        super().__init__(normalize_data=normalize_data, random_state=random_state)
        # save init args for serialization purposes
        self._init_args = {k: v for k, v in locals().items() if k in inspect.signature(self.__init__).parameters.keys()}

        """  ------ Setup model ------ """
        self.input_dim = input_dim

        if kernel_type == LatentKernel.RBF:
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.input_dim)
        elif kernel_type == LatentKernel.MATERN52:
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        elif kernel_type == LatentKernel.MATERN32:
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel_type == LatentKernel.MATERN12:
            base_kernel = gpytorch.kernels.MaternKernel(nu=0.5)
        else:
            raise ValueError(f"Unknown kernel type {kernel_type}")

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        self.covar_module.outputscale = kernel_variance
        self.covar_module.base_kernel.lengthscale = kernel_lengthscale

        self.mean_module = gpytorch.means.ZeroMean()

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = likelihood_std**2

        """ ------- normalization stats & data setup ------- """
        self._set_normalization_stats(normalization_stats)
        self.reset_to_prior()

    @property
    def kernel_lengthscale(self) -> float:
        return float(np.mean(self.covar_module.base_kernel.lengthscale.detach().numpy()))

    @property
    def kernel_outputscale(self) -> float:
        return float(self.covar_module.outputscale)

    @property
    def likelihood_std(self) -> float:
        return float(self.likelihood.noise**0.5)

    @property
    def gp(self) -> Callable:
        if self.X_data.shape[0] > 0:
            return self._gp
        else:
            return self._prior

    def _reset_posterior(self):
        x_context = torch.from_numpy(self.X_data)
        y_context = torch.from_numpy(self.y_data)
        self._gp = LearnedGPRegressionModel(
            x_context,
            y_context,
            self.likelihood,
            learned_kernel=None,
            learned_mean=None,
            covar_module=self.covar_module,
            mean_module=self.mean_module,
        )
        self._gp.eval()
        self.likelihood.eval()

    def _prior(self, x: np.ndarray):
        mean_x = self.mean_module(x).squeeze()
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def reset_to_prior(self):
        self._reset_data()
        self._gp = None

    def predict(self, test_x: np.ndarray, return_density: bool = False, include_obs_noise: bool = True, **kwargs):
        """
        computes the predictive distribution of the targets p(t|test_x, train_x, train_y)

        Args:
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density (bool) whether to return a density object or

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(y_test|X_test, X_train, y_train)
        """
        if test_x.ndim == 1:
            test_x = np.expand_dims(test_x, axis=-1)

        with torch.no_grad():
            test_x_normalized = self._normalize_data(test_x)
            test_x_tensor = torch.from_numpy(test_x_normalized)

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

    def pred_mean_grad(self, test_x: np.ndarray) -> np.ndarray:
        if test_x.ndim == 1:
            test_x = np.expand_dims(test_x, axis=-1)
        test_x_normalized = self._normalize_data(test_x)
        test_x_tensor = torch.tensor(test_x_normalized, requires_grad=True)
        pred_dist = self.gp(test_x_tensor)
        pred_dist_transformed = AffineTransformedDistribution(
            pred_dist, normalization_mean=self.y_mean, normalization_std=self.y_std
        )
        pred_mean = pred_dist_transformed.mean.sum()
        pred_mean.backward()
        pred_mean_grad_x = test_x_tensor.grad.detach().numpy() * self.x_std[None, :]
        assert pred_mean_grad_x.shape == test_x_normalized.shape
        return pred_mean_grad_x

    def _vectorize_pred_dist(self, pred_dist):
        return torch.distributions.Normal(pred_dist.mean, pred_dist.stddev)

    def meta_calibration_sharpness(
        self,
        x_context: np.ndarray,
        y_context: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        min_conf_level: float = 0.8,
        num_conf_levels: int = 20,
    ) -> Tuple[float, float]:
        # gp inference
        x_context, y_context = self._normalize_data(*self._handle_input_dim(x_context, y_context))
        x_context, y_context = torch.from_numpy(x_context), torch.from_numpy(y_context)

        test_x = self._normalize_data(X=x_test, Y=None)
        test_x = torch.from_numpy(test_x)

        with torch.no_grad():
            # compute posterior given the context data
            gp_model = LearnedGPRegressionModel(
                x_context, y_context, self.likelihood, covar_module=self.covar_module, mean_module=self.mean_module
            )
            gp_model.eval()
            self.likelihood.eval()
            pred_dist = self.likelihood(gp_model(test_x))
            pred_dist_transformed = AffineTransformedDistribution(
                pred_dist, normalization_mean=self.y_mean, normalization_std=self.y_std
            )

            pred_mean, pred_std = pred_dist_transformed.mean.numpy(), pred_dist_transformed.stddev.numpy()
        return self._calibration_sharpness(
            pred_mean, pred_std, y_test, min_conf_level=min_conf_level, num_conf_levels=num_conf_levels
        )

    def meta_calibration_sharpness_for_dataset(
        self, x: np.ndarray, y: np.ndarray, min_conf_level: float = 0.8
    ) -> Tuple[float, float]:
        result_list = [
            self.meta_calibration_sharpness(x[:i], y[:i], x[i:], y[i:], min_conf_level=min_conf_level)
            for i in range(2, x.shape[0])
        ]
        calibrated = float(np.mean(list(zip(*result_list))[0]))
        avg_std = float(np.mean(list(zip(*result_list))[1]))
        return calibrated, avg_std

    @classmethod
    def select_hparam_via_bracket_cv(
        cls,
        default_kwargs: Dict,
        meta_train_data: List[Tuple[np.ndarray, np.ndarray]],
        target_param: str = "kernel_lengthscale",
        min_calib_freq: float = 0.995,
        logspace: bool = True,
        upper: float = 3.0,
        lower: float = 0.1,
        num_iters: int = 5,
        verbose: bool = True,
        min_conf_level=0.8,
        increase_when_uncalibrated: bool = False,
    ) -> RegressionModel:
        assert 0.0 < min_calib_freq <= 1.0

        def calib_sharpness(gp_kwargs, x_data_test, y_data_test):
            gp = cls(**gp_kwargs)
            return gp.meta_calibration_sharpness_for_dataset(x_data_test, y_data_test, min_conf_level=min_conf_level)

        transform_fn = np.log10 if logspace else lambda x: x
        inv_transform_fn = lambda x: 10**x if logspace else lambda x: x
        lower = transform_fn(lower)
        upper = transform_fn(upper)

        for i in range(num_iters):
            middle = (upper + lower) / 2.0
            param = inv_transform_fn(middle)
            gp_kwargs = copy.deepcopy(default_kwargs)
            gp_kwargs[target_param] = param
            calibr_sharpness_results = [
                calib_sharpness(gp_kwargs, x_data, y_data) for x_data, y_data in meta_train_data
            ]
            calib_freq = np.mean(list(zip(*calibr_sharpness_results))[0])
            avg_std = np.mean(list(zip(*calibr_sharpness_results))[1])
            if verbose:
                print(
                    f"iter {i}/{num_iters} | upper: {10**upper} | lower = {10**lower} "
                    f"| {target_param}: {param} | calib_freq = {calib_freq} | avg_std = {avg_std}"
                )

            if (calib_freq >= min_calib_freq) ^ increase_when_uncalibrated:
                lower = middle
            else:
                upper = middle

        final_param = inv_transform_fn(upper) if increase_when_uncalibrated else inv_transform_fn(lower)
        gp_kwargs = copy.deepcopy(default_kwargs)
        gp_kwargs[target_param] = final_param
        if verbose:
            print(f"Final parameter {target_param} chosen: {final_param}")
        return cls(**gp_kwargs)

    @classmethod
    def select_kernel_via_frontier_cv(
        cls,
        default_kwargs: Dict,
        meta_train_data: List[Tuple[np.ndarray, np.ndarray]],
        min_calib_freq: float = 0.9,
        logspace: bool = True,
        upper_lengthscale: float = 5.0,
        lower_lengthscale: float = 0.1,
        upper_variance: float = 2.0,
        lower_variance: float = 0.9,
        min_conf_level: float = 0.8,
        num_iters: int = 50,
        num_data_permutations: int = 1,
        solver_v2: bool = True,
        verbose: bool = True,
    ) -> RegressionModel:
        assert 0.0 < min_calib_freq <= 1.0

        def calib_sharpness(gp_kwargs, x_data_test, y_data_test):
            gp = cls(**gp_kwargs)
            return gp.meta_calibration_sharpness_for_dataset(x_data_test, y_data_test, min_conf_level=min_conf_level)

        if solver_v2:
            transform_fn_variance = np.log10 if logspace else lambda x: x
            inv_transform_fn_variance = lambda x: 10**x if logspace else lambda x: x
            transform_fn_ls = lambda x: -transform_fn_variance(x)
            inv_transform_fn_ls = lambda x: inv_transform_fn_variance(-x)

            lower = np.array([transform_fn_variance(lower_variance), transform_fn_ls(upper_lengthscale)])
            upper = np.array([transform_fn_variance(upper_variance), transform_fn_ls(lower_lengthscale)])

            optim = MonotoneFrontierSolverV2(ndim=2, lower_boundary=lower, upper_boundary=upper)
        else:
            transform_fn_ls = np.log10 if logspace else lambda x: x
            inv_transform_fn_ls = lambda x: 10**x if logspace else lambda x: x
            transform_fn_variance = lambda x: -transform_fn_ls(x)
            inv_transform_fn_variance = lambda x: inv_transform_fn_ls(-x)

            lower = np.array([transform_fn_variance(upper_variance), transform_fn_ls(lower_lengthscale)])
            upper = np.array([transform_fn_variance(lower_variance), transform_fn_ls(upper_lengthscale)])
            optim = MonotoneFrontierSolver(ndim=2, lower_boundary=lower, upper_boundary=upper)

        if num_data_permutations == 1:
            datasets = meta_train_data
        elif num_data_permutations == 2:
            datasets = []
            for x, y in meta_train_data:
                datasets.append((x, y))
                datasets.append((np.flip(x, axis=0), np.flip(y, axis=0)))
        else:
            raise NotImplementedError("At the moment only num_data_permutations = 1, 2 is implemented")

        for i in range(num_iters):
            query_point = optim.next()
            kernel_variance = inv_transform_fn_variance(query_point[0])
            kernel_lengthscale = inv_transform_fn_ls(query_point[1])
            gp_kwargs = copy.deepcopy(default_kwargs)
            gp_kwargs["kernel_variance"] = kernel_variance
            gp_kwargs["kernel_lengthscale"] = kernel_lengthscale
            calibr_sharpness_results = [calib_sharpness(gp_kwargs, x_data, y_data) for x_data, y_data in datasets]
            calib_freq = float(np.mean(list(zip(*calibr_sharpness_results))[0]))
            avg_std = float(np.mean(list(zip(*calibr_sharpness_results))[1]))

            _, _, avg_std_best = optim.best_safe_evaluation
            if verbose:
                print(
                    f"iter {i}/{num_iters} | kernel_variance {kernel_variance} | kernel_lengthscale {kernel_lengthscale} "
                    f"| calib_freq = {calib_freq} (min {min_calib_freq}) | avg_std = {avg_std} | best avg_std so far: {avg_std_best}"
                )

            if solver_v2:
                optim.add_eval(point=query_point, constr=calib_freq - min_calib_freq, objective=avg_std)
            else:
                optim.add_eval(point=query_point, constr=min_calib_freq - calib_freq, objective=avg_std)

        param_best, _, avg_std_best = optim.best_safe_evaluation
        kernel_variance = inv_transform_fn_variance(param_best[0])
        kernel_lengthscale = inv_transform_fn_ls(param_best[1])
        gp_kwargs = copy.deepcopy(default_kwargs)
        gp_kwargs["kernel_variance"] = kernel_variance
        gp_kwargs["kernel_lengthscale"] = kernel_lengthscale
        if verbose:
            print(
                f"Final parameter chosen (with avg_std: {avg_std_best}): "
                f"variance {kernel_variance} | lengthscale {kernel_lengthscale} "
            )
        return cls(**gp_kwargs)


if __name__ == "__main__":
    import torch
    import numpy as np
    from matplotlib import pyplot as plt

    n_train_samples = 20
    n_test_samples = 200

    torch.manual_seed(25)
    x_data = torch.normal(mean=-1, std=2.0, size=(n_train_samples + n_test_samples, 1))
    W = torch.tensor([[0.6]])
    b = torch.tensor([-1])
    y_data = (
        x_data.matmul(W.T)
        + torch.sin((0.6 * x_data) ** 2)
        + b
        + torch.normal(mean=0.0, std=0.1, size=(n_train_samples + n_test_samples, 1))
    )
    y_data = torch.reshape(y_data, (-1,))

    x_data_train, x_data_test = x_data[:n_train_samples].numpy(), x_data[n_train_samples:].numpy()
    y_data_train, y_data_test = y_data[:n_train_samples].numpy(), y_data[n_train_samples:].numpy()

    gp_mll = GPRegressionVanilla(input_dim=x_data.shape[-1], kernel_lengthscale=1.0)
    gp_mll.add_data(x_data_train, y_data_train)

    x_plot = np.linspace(6, -6, num=200)
    gp_mll.confidence_intervals(x_plot)

    pred_mean, pred_std = gp_mll.predict(x_plot)
    pred_mean, pred_std = pred_mean.flatten(), pred_std.flatten()

    pred_mean_grad = gp_mll.pred_mean_grad(x_plot)
    plt.plot(x_plot, pred_mean_grad)

    plt.scatter(x_data_test, y_data_test)
    plt.plot(x_plot, pred_mean)

    # lcb, ucb = pred_mean - pred_std, pred_mean + pred_std
    lcb, ucb = gp_mll.confidence_intervals(x_plot)
    plt.fill_between(x_plot, lcb, ucb, alpha=0.4)
    plt.show()
