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

from typing import Optional, Tuple
from alef.models.base_model import BaseModel
import numpy as np
from ahgp.inference.hyperparam import hyperparam
from ahgp.gp.gp_helper import cal_kern_spec_mix_nomu_sep, GP_noise, standardize, cal_marg_likelihood_single
import torch
from scipy.stats import norm


class AHGPModel(BaseModel):
    def __init__(self, model_config_path: str, **args):
        super().__init__()
        self.model_config_path = model_config_path
        self.current_x_data = None
        self.current_y_data = None
        self.hyper_params = None

    def infer(self, x_data: np.array, y_data: np.array):
        y_data = np.squeeze(y_data)
        self.current_x_data = x_data
        self.current_y_data = y_data
        x_t, _, _ = standardize(x_data)
        x_t = x_t * 0.1
        y_t, _, _ = standardize(y_data)
        self.hyper_params = hyperparam(x_t, y_t, x_t, self.model_config_path, use_gpu=False)

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        assert self.current_y_data is not None
        assert self.current_x_data is not None
        device = torch.device("cpu")
        epsilon = 1.0e-2
        x_t, x_v, _, _ = standardize(self.current_x_data, x_test)
        x_t = x_t * 0.1
        x_v = x_v * 0.1
        y_t, mean_y_train, std_y_train = standardize(self.current_y_data)
        train_x = torch.from_numpy(x_t).float()
        train_y = torch.from_numpy(y_t).float().unsqueeze(-1)
        test_x = torch.from_numpy(x_v).float()
        var = self.hyper_params["var"]
        weights = self.hyper_params["weights"]
        K11 = cal_kern_spec_mix_nomu_sep(train_x, train_x, var, weights)
        K12 = cal_kern_spec_mix_nomu_sep(train_x, test_x, var, weights)
        K22 = cal_kern_spec_mix_nomu_sep(test_x, test_x, var, weights)
        -cal_marg_likelihood_single(K11, train_y, epsilon, device)
        mu_test, var_test = GP_noise(train_y, K11, K12, K22, epsilon, device)
        mu_test = mu_test.detach().squeeze(-1).cpu().numpy()
        var_test = var_test.detach().squeeze(-1).cpu().numpy().diagonal()
        # mu_test, var_test = predict(x_t, y_t, x_v, self.model_config_path, use_gpu=False)
        mu_test = mu_test * std_y_train + mean_y_train
        var_test = var_test * std_y_train**2
        return mu_test, np.sqrt(var_test)

    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        pred_mus, pred_sigmas = self.predictive_dist(x_test)
        log_likelis = norm.logpdf(np.squeeze(y_test), np.squeeze(pred_mus), np.squeeze(pred_sigmas))
        return log_likelis

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        assert self.current_y_data is not None
        assert self.current_x_data is not None
        raise NotImplementedError

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> np.float:
        raise NotImplementedError

    def reset_model(self):
        pass
