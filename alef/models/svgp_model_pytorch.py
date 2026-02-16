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
from alef.enums.global_model_enums import PredictionQuantity
from alef.models.base_model import BaseModel
import gpytorch
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from scipy.stats import norm


class GPModel(ApproximateGP):
    def __init__(self, inducing_points, mean_function, kernel_module):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = mean_function
        self.kernel_module = kernel_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.kernel_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVGPModelPytorch(BaseModel):
    def __init__(
        self,
        kernel: gpytorch.kernels.Kernel,
        initial_likelihood_noise: float,
        fix_likelihood_variance: bool,
        add_constant_mean_function: bool,
        prediction_quantity: PredictionQuantity,
        n_epochs: int,
        batch_size: int,
        lr: float,
        n_inducing_points: int,
        use_fraction_for_inducing_points: bool,
        fraction_inducing_points: float,
        batch_size_is_dataset_size: bool,
        **kwargs,
    ) -> None:
        super().__init__()
        self.kernel_module = kernel
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = torch.tensor(np.power(initial_likelihood_noise, 2.0))
        if fix_likelihood_variance:
            self.likelihood.noise_covar.raw_noise.requires_grad_(False)
        else:
            self.likelihood.noise_covar.raw_noise.requires_grad_(True)
        if add_constant_mean_function:
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            self.mean_module = gpytorch.means.ZeroMean()
        self.prediction_quantity = prediction_quantity
        self.model = None
        self.n_epochs = n_epochs
        self.min_epochs = 10
        self.batch_size = batch_size
        self.batch_size_is_dataset_size = batch_size_is_dataset_size
        self.lr = lr
        self.use_fraction_for_inducing_points = use_fraction_for_inducing_points
        self.fraction_inducing_points = fraction_inducing_points
        self.n_inducing_points = n_inducing_points

    def build_model(self, x_data: torch.tensor):
        if self.use_fraction_for_inducing_points:
            n_inducing_points = int(self.fraction_inducing_points * len(x_data))
        else:
            n_inducing_points = self.n_inducing_points
        if len(x_data) <= n_inducing_points:
            inducing_points = x_data
        else:
            inducing_points = x_data[:n_inducing_points]
        print(f"Num inducing points: {n_inducing_points}")
        self.model = GPModel(inducing_points, self.mean_module, self.kernel_module)

    def infer(self, x_data: np.array, y_data: np.array):
        x_data = torch.from_numpy(x_data).float()
        y_data = torch.from_numpy(np.squeeze(y_data)).float()
        self.build_model(x_data)
        self.maximize_elbo(x_data, y_data)

    def maximize_elbo(self, x_data, y_data):
        train_dataset = TensorDataset(x_data, y_data)
        if self.batch_size_is_dataset_size:
            train_loader = DataLoader(train_dataset, batch_size=len(x_data), shuffle=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=self.lr,
        )
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=y_data.size(0))
        for i in range(self.n_epochs):
            batch_counter = 0
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = -mll(output, y_batch)
                # print(f"Epoch {i}/{self.n_epochs} Batch no {batch_counter} - loss: {loss.item()}")
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()
                batch_counter += 1
            print(f"Epoch {i}/{self.n_epochs} - loss: {epoch_loss}")
            if i > self.min_epochs:
                if np.allclose(epoch_loss, previous_epoch_loss, rtol=1e-5, atol=1e-5):  # noqa: F821
                    print(f"Stopping criteria triggered at iteration {i}")
                    break
            previous_epoch_loss = epoch_loss  # noqa: F841

    def predict(self, x_test: torch.tensor):
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            if self.prediction_quantity == PredictionQuantity.PREDICT_F:
                f_pred = self.model(x_test)
                return f_pred
            elif self.prediction_quantity == PredictionQuantity.PREDICT_Y:
                y_pred = self.likelihood(self.model(x_test))
                return y_pred

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        x_test = torch.from_numpy(x_test).float()
        pred_dist = self.predict(x_test)
        mean = pred_dist.mean.numpy()
        covar_matrix = pred_dist.covariance_matrix.detach().numpy()
        std = np.sqrt(np.diag(covar_matrix))
        assert len(mean) == len(x_test) and len(std) == len(x_test)
        return mean, std

    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        assert self.prediction_quantity == PredictionQuantity.PREDICT_Y
        pred_mu, pred_std = self.predictive_dist(x_test)
        log_likelis = norm.logpdf(np.squeeze(y_test), np.squeeze(pred_mu), np.squeeze(pred_std))
        return log_likelis

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> np.float:
        raise NotImplementedError

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        raise NotImplementedError

    def reset_model(self):
        pass
