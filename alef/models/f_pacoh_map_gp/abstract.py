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
import torch
import scipy.stats
import copy
from alef.models.f_pacoh_map_gp.util import _handle_input_dimensionality
from typing import Tuple, Dict, List

"""
This is from the attached code of paper
Jonas Rothfuss, Christopher Koenig, Alisa Rupenyan, Andreas Krause, CoRL 2022,
Meta-Learning Priors for Safe Bayesian Optimization

The link is published by the authors
https://tinyurl.com/safe-meta-bo

see folder ./meta_bo/models/abstract.py

Copyright (c) 2022 Jonas Rothfuss, licensed under the MIT License

"""


class RegressionModel:
    def __init__(self, normalize_data=True, random_state=None):
        self.normalize_data = normalize_data
        self.input_dim = None

        self._rds = random_state if random_state is not None else np.random
        torch.manual_seed(self._rds.randint(0, 10**7))

    def predict(self, test_x: np.ndarray, return_density: bool = False, include_obs_noise: bool = True, **kwargs):
        raise NotImplementedError

    def eval(self, test_x, test_y, **kwargs):
        """
        Computes the average test log likelihood and the rmse on test data

        Args:
            test_x: (ndarray) test input data of shape (n_samples, ndim_x)
            test_y: (ndarray) test target data of shape (n_samples, 1)

        Returns: (avg_log_likelihood, rmse)

        """
        # convert to tensors
        test_x, test_y = _handle_input_dimensionality(test_x, test_y)
        test_t_tensor = torch.from_numpy(test_y).contiguous().float().flatten()

        with torch.no_grad():
            pred_dist = self.predict(test_x, return_density=True, *kwargs)
            avg_log_likelihood = pred_dist.log_prob(test_t_tensor) / test_t_tensor.shape[0]
            rmse = torch.mean(torch.pow(pred_dist.mean - test_t_tensor, 2)).sqrt()

            pred_dist_vect = self._vectorize_pred_dist(pred_dist)
            calibr_error = self._calib_error(pred_dist_vect, test_t_tensor)
            calibr_error_chi2 = _calib_error_chi2(pred_dist_vect, test_t_tensor)

            return avg_log_likelihood.cpu().item(), rmse.cpu().item(), calibr_error.cpu().item(), calibr_error_chi2

    def confidence_intervals(self, test_x, confidence=0.9, **kwargs):
        pred_dist = self.predict(test_x, return_density=True, **kwargs)
        pred_dist = self._vectorize_pred_dist(pred_dist)

        alpha = (1 - confidence) / 2
        ucb = pred_dist.icdf(torch.ones(test_x.size) * (1 - alpha))
        lcb = pred_dist.icdf(torch.ones(test_x.size) * alpha)
        return ucb, lcb

    def _reset_posterior(self):
        raise NotImplementedError

    def _reset_data(self):
        self.X_data = torch.empty(size=(0, self.input_dim), dtype=torch.float64)
        self.y_data = torch.empty(size=(0,), dtype=torch.float64)
        self._num_train_points = 0

    def _handle_input_dim(self, X, y):
        if X.ndim == 1:
            assert X.shape[-1] == self.input_dim
            X = X.reshape((-1, self.input_dim))

        if isinstance(y, float) or y.ndim == 0:
            y = np.array(y)
            y = y.reshape((1,))
        elif y.ndim == 1:
            pass
        else:
            raise AssertionError("y must not have more than 1 dim")
        return X, y

    def add_data(self, X, y):
        assert X.ndim == 1 or X.ndim == 2

        # handle input dimensionality
        X, y = self._handle_input_dim(X, y)

        # normalize data
        X, y = self._normalize_data(X, y)
        y = y.flatten()

        if self._num_train_points == 0 and y.shape[0] == 1:
            # for some reason gpytorch can't deal with one data point
            # thus store first point double and remove later
            self.X_data = np.concatenate([self.X_data, X])
            self.y_data = np.concatenate([self.y_data, y])
        if self._num_train_points == 1 and self.X_data.shape[0] == 2:
            # remove duplicate datapoint
            self.X_data = self.X_data[:1, :]
            self.y_data = self.y_data[:1]

        self.X_data = np.concatenate([self.X_data, X])
        self.y_data = np.concatenate([self.y_data, y])

        self._num_train_points += y.shape[0]

        assert self.X_data.shape[0] == self.y_data.shape[0]
        assert self._num_train_points == 1 or self.X_data.shape[0] == self._num_train_points

        self._reset_posterior()

    def remove_last_data_point(self):
        assert self.X_data.ndim == 2
        self.X_data = self.X_data[:-1]
        self.y_data = self.y_data[:-1]
        self._num_train_points -= 1
        self._reset_posterior()

    def _set_normalization_stats(self, normalization_stats_dict=None):
        if normalization_stats_dict is None:
            self.x_mean, self.y_mean = np.zeros(self.input_dim), np.zeros(1)
            self.x_std, self.y_std = np.ones(self.input_dim), np.ones(1)
        else:
            self.x_mean = normalization_stats_dict["x_mean"].reshape((self.input_dim,))
            self.y_mean = normalization_stats_dict["y_mean"].squeeze()
            self.x_std = normalization_stats_dict["x_std"].reshape((self.input_dim,))
            self.y_std = normalization_stats_dict["y_std"].squeeze()

    def _calib_error(self, pred_dist_vectorized, test_t_tensor):
        return _calib_error(pred_dist_vectorized, test_t_tensor)

    @staticmethod
    def _calibration_sharpness(
        pred_mean, pred_std, test_y: np.ndarray, min_conf_level: float = 0.5, num_conf_levels: int = 20
    ) -> Tuple[float, float]:
        # compute what percentage of the tested uncertainty estimates are calibrated and how sharp they are on average
        assert 0.0 <= min_conf_level < 1.0
        conf_levels = np.linspace(min_conf_level, 1, num_conf_levels)
        z2 = ((pred_mean - test_y.flatten()) / pred_std) ** 2
        z2_threshold = scipy.stats.chi2.ppf(conf_levels, 1)
        empirical_coverage = np.mean(z2 <= z2_threshold[:, None], axis=-1)

        calibrated_avg = float(np.mean(empirical_coverage >= conf_levels))
        avg_std = float(np.mean(pred_std))
        return calibrated_avg, avg_std

    def calibration_sharpness(
        self, test_x: np.ndarray, test_y: np.ndarray, min_conf_level: float = 0.5, num_conf_levels: int = 20
    ) -> Tuple[float, float]:
        # compute whether uncertainty estimates are calibrated and sharp
        pred_mean, pred_std = self.predict(test_x, return_density=False)
        return self._calibration_sharpness(
            pred_mean, pred_std, test_y, min_conf_level=min_conf_level, num_conf_levels=num_conf_levels
        )

    def _compute_normalization_stats(self, X, Y):
        # save mean and variance of data for normalization
        if self.normalize_data:
            self.x_mean, self.y_mean = np.mean(X, axis=0), np.mean(Y, axis=0)
            self.x_std, self.y_std = np.std(X, axis=0) + 1e-8, np.std(Y, axis=0) + 1e-8
        else:
            self.x_mean, self.y_mean = np.zeros(X.shape[1]), np.zeros(Y.shape[1])
            self.x_std, self.y_std = np.ones(X.shape[1]), np.ones(Y.shape[1])

    def _normalize_data(self, X, Y=None):
        assert hasattr(self, "x_mean") and hasattr(self, "x_std"), "requires computing normalization stats beforehand"
        assert hasattr(self, "y_mean") and hasattr(self, "y_std"), "requires computing normalization stats beforehand"

        X_normalized = (X - self.x_mean[None, :]) / self.x_std[None, :]

        if Y is None:
            return X_normalized
        else:
            Y_normalized = (Y - self.y_mean) / self.y_std
            return X_normalized, Y_normalized

    def _unnormalize_pred(self, pred_mean, pred_std):
        assert hasattr(self, "x_mean") and hasattr(self, "x_std"), "requires computing normalization stats beforehand"
        assert hasattr(self, "y_mean") and hasattr(self, "y_std"), "requires computing normalization stats beforehand"

        if self.normalize_data:
            assert pred_mean.ndim == pred_std.ndim == 2 and pred_mean.shape[1] == pred_std.shape[1] == self.output_dim
            if isinstance(pred_mean, torch.Tensor) and isinstance(pred_std, torch.Tensor):
                y_mean_tensor, y_std_tensor = torch.tensor(self.y_mean).float(), torch.tensor(self.y_std).float()
                pred_mean = pred_mean.mul(y_std_tensor[None, :]) + y_mean_tensor[None, :]
                pred_std = pred_std.mul(y_std_tensor[None, :])
            else:
                pred_mean = pred_mean.multiply(self.y_std[None, :]) + self.y_mean[None, :]
                pred_std = pred_std.multiply(self.y_std[None, :])

        return pred_mean, pred_std

    def _initial_data_handling(self, train_x, train_t):
        train_x, train_t = _handle_input_dimensionality(train_x, train_t)
        self.input_dim, self.output_dim = train_x.shape[-1], train_t.shape[-1]
        self.n_train_samples = train_x.shape[0]

        # b) normalize data to exhibit zero mean and variance
        self._compute_normalization_stats(train_x, train_t)
        train_x_normalized, train_t_normalized = self._normalize_data(train_x, train_t)

        # c) Convert the data into pytorch tensors
        self.train_x = torch.from_numpy(train_x_normalized).contiguous().float()
        self.train_t = torch.from_numpy(train_t_normalized).contiguous().float()

        return self.train_x, self.train_t

    def _vectorize_pred_dist(self, pred_dist):
        raise NotImplementedError

    @property
    def _torch_module_variables(self):
        return [var_name for var_name, val in self.__dict__.items() if isinstance(val, torch.nn.Module)]

    def __getstate__(self):
        state_dict = copy.copy(self.__dict__)
        _torch_module_names = self._torch_module_variables
        for module_name in _torch_module_names:
            if isinstance(state_dict[module_name], torch.nn.Module):
                state_dict[f"{module_name}_state"] = state_dict.pop(module_name).state_dict()
        state_dict["_torch_module_names"] = _torch_module_names
        state_dict["_rds_state"] = state_dict.pop("_rds").get_state()
        return state_dict

    def __setstate__(self, state):
        self.__init__(**state["_init_args"])
        self._rds.set_state(state.pop("_rds_state"))
        torch_states = {
            module_name: state.pop(f"{module_name}_state") for module_name in state.pop("_torch_module_names")
        }
        self.__dict__.update(state)
        if self.X_data.shape[0] > 0:
            self._reset_posterior()
        for module_name, state in torch_states.items():
            getattr(self, module_name).load_state_dict(state)


class RegressionModelMetaLearned(RegressionModel):
    def meta_fit(self, meta_train_tuples, meta_valid_tuples=None, verbose=True, **kwargs):
        raise NotImplementedError

    def meta_predict(
        self, context_x: np.ndarray, context_y: np.ndarray, test_x: np.ndarray, return_density: bool = False
    ):
        raise NotImplementedError

    def meta_calibration_sharpness(
        self,
        context_x: np.ndarray,
        context_y: np.ndarray,
        test_x: np.ndarray,
        test_y: np.ndarray,
        min_conf_level: float = 0.5,
        num_conf_levels: int = 20,
    ) -> Tuple[float, float]:
        pred_mean, pred_std = self.meta_predict(context_x, context_y, test_x, return_density=False)
        return self._calibration_sharpness(
            pred_mean, pred_std, test_y, min_conf_level=min_conf_level, num_conf_levels=num_conf_levels
        )

    def meta_calibration_sharpness_for_dataset(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[float, float]:
        result_list = [
            self.meta_calibration_sharpness(x[:i], y[:i], x[i:], y[i:], **kwargs) for i in range(2, x.shape[0])
        ]
        calib_freq = float(np.mean(list(zip(*result_list))[0]))
        avg_std = float(np.mean(list(zip(*result_list))[1]))
        return calib_freq, avg_std

    def meta_calibration_sharpness_for_datasets(
        self, meta_data_tuples: List[Tuple[np.ndarray, np.ndarray]], **kwargs
    ) -> Tuple[float, float]:
        calibr_sharpness_results = [
            self.meta_calibration_sharpness_for_dataset(x, y, **kwargs) for x, y in meta_data_tuples
        ]
        calib_freq = float(np.mean(list(zip(*calibr_sharpness_results))[0]))
        avg_std = float(np.mean(list(zip(*calibr_sharpness_results))[1]))
        return calib_freq, avg_std

    def eval_datasets(self, test_tuples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], **kwargs):
        """
        Performs meta-testing on multiple tasks / datasets.
        Computes the average test log likelihood, the rmse and the calibration error over multiple test datasets

        Args:
            test_tuples: list of test set tuples, i.e. [(test_context_x_1, test_context_y_1, test_x_1, test_y_1), ...]

        Returns: (avg_log_likelihood, rmse, calibr_error)

        """

        assert all([len(valid_tuple) == 4 for valid_tuple in test_tuples])

        ll_list, rmse_list, calibr_err_list, calibr_err_chi2_list = list(
            zip(*[self.meta_eval(*test_data_tuple, **kwargs) for test_data_tuple in test_tuples])
        )

        return np.mean(ll_list), np.mean(rmse_list), np.mean(calibr_err_list), np.mean(calibr_err_chi2_list)

    def meta_eval(self, context_x: np.ndarray, context_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray):
        test_x, test_y = _handle_input_dimensionality(test_x, test_y)
        test_y_tensor = torch.from_numpy(test_y).contiguous().float().flatten()

        with torch.no_grad():
            pred_dist = self.meta_predict(context_x, context_y, test_x, return_density=True)
            avg_log_likelihood = pred_dist.log_prob(test_y_tensor) / test_y_tensor.shape[0]
            rmse = torch.mean(torch.pow(pred_dist.mean - test_y_tensor, 2)).sqrt()

            pred_dist_vect = self._vectorize_pred_dist(pred_dist)
            calibr_error = self._calib_error(pred_dist_vect, test_y_tensor)
            calibr_error_chi2 = _calib_error_chi2(pred_dist_vect, test_y_tensor)

            return avg_log_likelihood.cpu().item(), rmse.cpu().item(), calibr_error.cpu().item(), calibr_error_chi2

    def _compute_meta_normalization_stats(self, meta_train_tuples: List[Tuple[np.ndarray, np.ndarray]]):
        X_stack, Y_stack = list(
            zip(*[_handle_input_dimensionality(x_train, y_train) for x_train, y_train in meta_train_tuples])
        )
        X, Y = np.concatenate(X_stack, axis=0), np.concatenate(Y_stack, axis=0)

        if self.normalize_data:
            self.x_mean, self.y_mean = np.mean(X, axis=0), np.mean(Y, axis=0)
            self.x_std, self.y_std = np.std(X, axis=0) + 1e-8, np.std(Y, axis=0) + 1e-8
        else:
            self.x_mean, self.y_mean = np.zeros(X.shape[1]), np.zeros(Y.shape[1])
            self.x_std, self.y_std = np.ones(X.shape[1]), np.ones(Y.shape[1])

    def _check_meta_data_shapes(self, meta_train_data: List[Tuple[np.ndarray, np.ndarray]]):
        for i in range(len(meta_train_data)):
            meta_train_data[i] = _handle_input_dimensionality(*meta_train_data[i])
        self.input_dim = meta_train_data[0][0].shape[-1]
        self.output_dim = meta_train_data[0][1].shape[-1]

        assert all(
            [
                self.input_dim == train_x.shape[-1] and self.output_dim == train_t.shape[-1]
                for train_x, train_t in meta_train_data
            ]
        )

    def _prepare_data_per_task(self, x_data: np.ndarray, y_data: np.ndarray, flatten_y: bool = True):
        # a) make arrays 2-dimensional
        x_data, y_data = _handle_input_dimensionality(x_data, y_data)

        # b) normalize data
        x_data, y_data = self._normalize_data(x_data, y_data)

        if flatten_y:
            assert y_data.shape[1] == 1
            y_data = y_data.flatten()

        # c) convert to tensors
        x_tensor = torch.from_numpy(x_data).float()
        y_tensor = torch.from_numpy(y_data).float()

        return x_tensor, y_tensor

    @classmethod
    def select_hparam_via_bracket_cv(
        cls,
        default_kwargs: Dict,
        meta_train_data: List[Tuple[np.ndarray, np.ndarray]],
        target_param: str,
        min_calib_freq: float = 0.99,
        logspace: bool = True,
        upper: float = 1.0,
        lower: float = 1e-3,
        num_iters: int = 6,
        verbose: bool = True,
        increase_when_uncalibrated: bool = True,
    ) -> RegressionModel:
        assert 0.0 < min_calib_freq <= 1.0

        def calib_sharpness(gp_kwargs, meta_train_data, x_data_test, y_data_test):
            torch.set_num_threads(2)
            model = cls(**gp_kwargs)
            model.meta_fit(meta_train_data, log_period=20)
            return model.meta_calibration_sharpness_for_dataset(x_data_test, y_data_test)

        transform_fn = np.log10 if logspace else lambda x: x
        inv_transform_fn = lambda x: 10**x if logspace else lambda x: x
        lower = transform_fn(lower)
        upper = transform_fn(upper)

        for i in range(num_iters):
            middle = (upper + lower) / 2.0
            param = inv_transform_fn(middle)
            gp_kwargs = copy.deepcopy(default_kwargs)
            gp_kwargs[target_param] = param
            # do leave-one-out cross-validation on the meta-training tasks
            calibr_sharpness_results = [
                calib_sharpness(gp_kwargs, meta_train_data[:i] + meta_train_data[i + 1 :], *meta_train_data[i])
                for i in range(len(meta_train_data))
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


def _calib_error(pred_dist_vectorized, test_t_tensor):
    cdf_vals = pred_dist_vectorized.cdf(test_t_tensor)

    if test_t_tensor.shape[0] == 1:
        test_t_tensor = test_t_tensor.flatten()
        cdf_vals = cdf_vals.flatten()

    num_points = test_t_tensor.shape[0]
    conf_levels = torch.linspace(0.05, 1.0, 20)
    emp_freq_per_conf_level = torch.sum(cdf_vals[:, None] <= conf_levels, dim=0).float() / num_points

    calib_rmse = torch.sqrt(torch.mean((emp_freq_per_conf_level - conf_levels) ** 2))
    return calib_rmse


def _calib_error_chi2(pred_dist_vectorized, test_t_tensor):
    import scipy.stats

    z2 = (((pred_dist_vectorized.mean - test_t_tensor) / pred_dist_vectorized.stddev) ** 2).detach().numpy()
    f = lambda p: np.mean(z2 < scipy.stats.chi2.ppf(p, 1))
    conf_levels = np.linspace(0.05, 1, 20)
    accs = np.array([f(p) for p in conf_levels])
    return np.sqrt(np.mean((accs - conf_levels) ** 2))
