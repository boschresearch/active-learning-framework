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

from typing import List, Optional, Tuple, Union
from alef.models.amortized_infer_structured_kernels.gp.base_kernels import (
    KernelTypeList,
    transform_kernel_list_to_expression,
)
from alef.models.base_model import BaseModel
import numpy as np
from alef.models.amortized_infer_structured_kernels.gp.base_symbols import BaseKernelTypes
from alef.models.bayesian_ensemble_interface import BayesianEnsembleInterface
from alef.models.gp_model_pytorch import GPModelPytorch
from alef.utils.gaussian_mixture_density import GaussianMixtureDensity


class GPModelPytorchEnsemble(BaseModel, BayesianEnsembleInterface):
    """
    Method for building a Bayesian Model Average over GP models with different kernel structures
    Here, kernel parameters are point estimates (no marginalization on parameter level)
    Kernels specfied via KernelTypeList elements and thus have the form of
    multiplication of kernels over dimensions and additions of base kernels inside dimensions
    """

    def __init__(self, gp_model_config, **kwargs):
        self.gp_model_config = gp_model_config
        self.ensemble_list: List[GPModelPytorch] = []
        self.ensemble_log_marginal_likelis = []

    def set_kernel_list(self, kernel_list: Union[List[List[BaseKernelTypes]], List[KernelTypeList]]):
        """
        Method for setting the ensemble in form of kernel structures - input can be either two layer nested list or three layer
        the first list are the ensemble elements - the second list specfies the kernel via KernelTypeList - if it is only one list here then
        this list is interpreted as the same kernel list applied to each dimension aka it is transformed to a KernelTypeList in self.create_input_kernel_list

        KernelTypeList specifies a kernel that is a multiplication of kernels over dimensions and additions of base kernels inside dimensions
        """
        assert isinstance(kernel_list[0], list)
        if isinstance(kernel_list[0][0], list):
            self.kernel_list_defined_on_all_dimensions = True
        else:
            self.kernel_list_defined_on_all_dimensions = False
        self.kernel_list = kernel_list
        self.n_ensemble = len(self.kernel_list)

    def create_input_kernel_list(
        self, kernel_list: Union[List[List[BaseKernelTypes]], List[KernelTypeList]], n_dims: int
    ) -> List[KernelTypeList]:
        if self.kernel_list_defined_on_all_dimensions:
            return kernel_list
        else:
            new_kernel_list = []
            for kernel_list_in_ensemble in kernel_list:
                kernel_list_over_dim = [kernel_list_in_ensemble for i in range(0, n_dims)]
                new_kernel_list.append(kernel_list_over_dim)
            return new_kernel_list

    def infer(self, x_data: np.array, y_data: np.array):
        """
        Builds a BMA over kernel structures with marginal likelihoods (normalized) as ensemble weights
        """
        self.ensemble_list = []
        self.ensemble_log_marginal_likelis = []
        n_dim = x_data.shape[1]
        n_data = x_data.shape[0]
        kernel_list = self.create_input_kernel_list(self.kernel_list, n_dim)
        n_kernels = len(kernel_list)
        for i, kernel_inner_list in enumerate(kernel_list):
            print(f"Process kernel {i + 1}/{n_kernels}:")
            print(kernel_inner_list)
            kernel_expression = transform_kernel_list_to_expression(kernel_inner_list, True)
            gp_model = GPModelPytorch(kernel_expression.get_kernel(), **self.gp_model_config.dict())
            gp_model.infer(x_data, y_data)
            log_marginal_likeli = gp_model.eval_log_marginal_likelihood(x_data, y_data).item() * n_data
            self.ensemble_log_marginal_likelis.append(log_marginal_likeli)
            self.ensemble_list.append(gp_model)

        ensemble_log_marginal_likelis = np.array(self.ensemble_log_marginal_likelis)
        max_log_marginal_likeli = np.max(ensemble_log_marginal_likelis)
        # subtract max log marignal likeli for numerical stability of softmax
        unnnormalized_ensemble_weights = np.exp(ensemble_log_marginal_likelis - max_log_marginal_likeli)
        self.ensemble_weights = unnnormalized_ensemble_weights / np.sum(unnnormalized_ensemble_weights)

    def predict(self, x_test: np.array) -> Tuple[np.array, np.array]:
        pred_mus = []
        pred_sigmas = []
        for i in range(0, self.n_ensemble):
            mu_test, sigma_test = self.ensemble_list[i].predictive_dist(x_test)
            pred_mus.append(mu_test)
            pred_sigmas.append(sigma_test)
        pred_mus = np.array(pred_mus)
        pred_sigmas = np.array(pred_sigmas)
        assert len(pred_sigmas.shape) == 2
        assert len(pred_mus.shape) == 2
        return pred_mus, pred_sigmas

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        """
        Method for retrieving the predictive mean and sigma for a given array of the test points

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        mean array with shape (n,)
        sigma array with shape (n,)
        """
        pred_mus_complete, pred_sigmas_complete = self.predict(x_test)
        print("Ensemble weights:")
        print(self.ensemble_weights)
        n = x_test.shape[0]
        mus_over_inputs = []
        sigmas_over_inputs = []
        for i in range(0, n):
            dist = GaussianMixtureDensity(self.ensemble_weights, pred_mus_complete[:, i], pred_sigmas_complete[:, i])
            mu = dist.mean()
            var = dist.variance()
            mus_over_inputs.append(mu)
            sigmas_over_inputs.append(np.sqrt(var))
        return np.array(mus_over_inputs), np.array(sigmas_over_inputs)

    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        pred_mus_complete, pred_sigmas_complete = self.predict(x_test)
        n = x_test.shape[0]
        log_likelis = []
        for i in range(0, n):
            gmm_at_test_point = GaussianMixtureDensity(
                self.ensemble_weights, pred_mus_complete[:, i], pred_sigmas_complete[:, i]
            )
            log_likeli = gmm_at_test_point.log_likelihood(np.squeeze(y_test[i]))
            log_likelis.append(log_likeli)
        return np.squeeze(np.array(log_likelis))

    def get_predictive_distributions(self, x_test: np.array) -> List[Tuple[np.array, np.array]]:
        pred_mus_complete, pred_sigmas_complete = self.predict(x_test)
        pred_dists = [
            (np.squeeze(pred_mus_complete[j, :]), np.squeeze(pred_sigmas_complete[j, :]))
            for j in range(0, self.n_ensemble)
        ]
        return pred_dists

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        return super().entropy_predictive_dist(x_test)

    def reset_model(self):
        return super().reset_model()

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> np.float:
        return super().estimate_model_evidence(x_data, y_data)
