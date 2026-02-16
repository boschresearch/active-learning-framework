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

import gpflow
from typing import List
from gpflow.utilities import print_summary
import numpy as np

f64 = gpflow.utilities.to_default_float


class NeuralKernelNetwork(gpflow.kernels.Kernel):
    def __init__(
        self,
        input_dimension: int,
        middle_layer_list: List[int],
        base_lengthscale: float,
        base_variance: float,
        base_period: float,
        base_alpha: float,
        init_linear_layer_a: float,
        init_linear_layer_b: float,
        **kwargs,
    ):
        super().__init__()
        self.init_linear_layer_a = init_linear_layer_a
        self.init_linear_layer_b = init_linear_layer_b
        rbf = gpflow.kernels.RBF(
            lengthscales=f64(np.repeat(base_lengthscale, input_dimension)), variance=f64(base_variance)
        )
        periodic = gpflow.kernels.Periodic(
            gpflow.kernels.RBF(
                lengthscales=f64(np.repeat(base_lengthscale, input_dimension)), variance=f64(base_variance)
            ),
            period=f64(base_period),
        )
        linear = gpflow.kernels.Linear(variance=f64(base_variance))
        rq = gpflow.kernels.RationalQuadratic(
            lengthscales=f64(np.repeat(base_lengthscale, input_dimension)),
            variance=f64(base_variance),
            alpha=f64(base_alpha),
        )
        rbf2 = gpflow.kernels.RBF(
            lengthscales=f64(np.repeat(base_lengthscale, input_dimension)), variance=f64(base_variance)
        )
        periodic2 = gpflow.kernels.Periodic(
            gpflow.kernels.RBF(
                lengthscales=f64(np.repeat(base_lengthscale, input_dimension)), variance=f64(base_variance)
            ),
            period=f64(base_period),
        )
        linear2 = gpflow.kernels.Linear(variance=f64(base_variance))
        rq2 = gpflow.kernels.RationalQuadratic(
            lengthscales=f64(np.repeat(base_lengthscale, input_dimension)),
            variance=f64(base_variance),
            alpha=f64(base_alpha),
        )
        base_kernels = [rbf, rbf2, periodic, periodic2, linear, linear2, rq, rq2]
        out = self.module(base_kernels, middle_layer_list[0])
        for i in range(1, len(middle_layer_list)):
            out = self.module(out, middle_layer_list[i])
        self.kernel = self.linear_layer(out, 1)[0]

    def linear_layer(self, kernel_list: List[gpflow.kernels.Kernel], n_outputs: int) -> List[gpflow.kernels.Kernel]:
        n_inputs = len(kernel_list)
        output_kernel_list = []
        for i in range(0, n_outputs):
            constant_kernel = gpflow.kernels.Constant(
                variance=np.random.uniform(self.init_linear_layer_a, self.init_linear_layer_b)
            )
            kernel = constant_kernel * kernel_list[0]
            for j in range(1, n_inputs):
                constant_kernel = gpflow.kernels.Constant(
                    variance=np.random.uniform(self.init_linear_layer_a, self.init_linear_layer_b)
                )
                kernel = kernel + constant_kernel * kernel_list[j]
            output_kernel_list.append(kernel)
        return output_kernel_list

    def product_layer(self, kernel_list: List[gpflow.kernels.Kernel]) -> List[gpflow.kernels.Kernel]:
        assert len(kernel_list) % 2 == 0
        n_kernels = len(kernel_list)
        output_kernel_list = []
        for i in range(0, n_kernels):
            if (i - 1) % 2 == 0:
                kernel = kernel_list[i] * kernel_list[i - 1]
                output_kernel_list.append(kernel)
        assert len(output_kernel_list) == (n_kernels / 2)
        return output_kernel_list

    def module(self, primitive_kernel_list: List[gpflow.kernels.Kernel], n_outputs: int) -> List[gpflow.kernels.Kernel]:
        output_linear_layer = self.linear_layer(primitive_kernel_list, n_outputs * 2)
        output_product_layer = self.product_layer(output_linear_layer)
        return output_product_layer

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return self.kernel.K(X, X2)

    def K_diag(self, X):
        return self.kernel.K_diag(X)


if __name__ == "__main__":
    nkn = NeuralKernelNetwork(2, 3)

    print_summary(nkn.kernel)
