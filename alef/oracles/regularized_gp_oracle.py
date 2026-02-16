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
from typing import Tuple
from alef.configs.models.gp_model_config import GPModelFastConfig
from alef.configs.kernels.invertible_resnet_kernel_configs import CurlRegularizedIResnetKernelConfig

# invertible_resnet_kernel_configs does not exist, please fix
from alef.data_sets.concrete import Concrete
from alef.models.model_factory import ModelFactory


class RegularizedGPOracle:
    def __init__(self) -> None:
        self.transform = True
        if self.transform:
            self.__a = -1.0
            self.__b = 0.5
        else:
            self.__a = 0.000001
            self.__b = 100.0

    def load_train_test_data_for_gp(self):
        data_loader = Concrete()
        data_loader.load_data_set()
        n_data = 50
        n_test = 400
        np.random.seed(109)
        self.x_data, self.y_data, self.x_test, self.y_test = data_loader.sample_train_test(
            use_absolute=True, n_train=n_data, n_test=n_test, fraction_train=None
        )

    def query(self, x: np.array) -> np.float:
        if self.transform:
            lambda_param = np.exp(10 * x[0])
        else:
            lambda_param = x[0]
        n_dim = self.x_data.shape[1]

        kernel_config = CurlRegularizedIResnetKernelConfig(
            input_dimension=n_dim,
            num_layers=20,
            layer_size_list=[20],
            regularizer_lambdas=[0.0, 0.0, 0.0, lambda_param],
            fix_lengthscales=False,
        )

        model_config = GPModelFastConfig(
            kernel_config=kernel_config,
            observation_noise=0.01,
            train_likelihood_variance=True,
            set_prior_on_observation_noise=False,
        )

        model_factory = ModelFactory()

        model = model_factory.build(model_config)

        model.infer(self.x_data, self.y_data)

        # pred_mu, pred_sigma = model.predictive_dist(self.x_test)

        # print(pred_mu.shape)

        ll = np.mean(model.predictive_log_likelihood(self.x_test, self.y_test))

        # rmse = calculate_rmse(pred_mu,self.y_test)

        return ll

    def get_random_data(self, n: int, noisy: bool = False) -> Tuple[np.array, np.array]:
        X = np.random.uniform(low=self.__a, high=self.__b, size=(n, self.get_dimension()))
        function_values = []
        for x in X:
            function_value = self.query(x)
            function_values.append(function_value)
        print("-Dataset of length " + str(n) + " generated")
        return X, np.expand_dims(np.array(function_values), axis=1)

    def get_box_bounds(self):
        return self.__a, self.__b

    def get_dimension(self):
        return 1


if __name__ == "__main__":
    mnist_svm = RegularizedGPOracle()
    mnist_svm.load_train_test_data_for_gp()
    print(mnist_svm.get_random_data(2))
