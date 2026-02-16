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

from gpflow.kernels import Kernel
import gpflow
import numpy as np

from typing import Tuple
from tensorflow_probability import distributions as tfd
from gpflow.utilities import print_summary

gpflow.config.set_default_float(np.float64)
f64 = gpflow.utilities.to_default_float


class Partition:
    def __init__(self, num_dims):
        self.partition_list = []
        self.num_dims = num_dims

    def set_partition_list(self, partition_list):
        self.partition_list = partition_list

    def add_partition_element(self, index_list):
        self.partition_list.append(index_list)

    def check_partition_validity(self):
        complete_index_list = []
        for partition in self.partition_list:
            for index in partition:
                complete_index_list.append(index)
        assert len(complete_index_list) == self.num_dims
        complete_index_list.sort()
        for i in range(0, self.num_dims):
            assert complete_index_list[i] == i

    def get_partition_for_index(self, index):
        return self.partition_list[index]

    def get_partition_list(self):
        return self.partition_list

    def get_num_dims(self):
        return self.num_dims


class AdditiveKernel(Kernel):
    def __init__(
        self,
        partition: Partition,
        base_lengthscale: float,
        base_variance: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        **kwargs,
    ):
        partition.check_partition_validity()
        partition_list = partition.get_partition_list()
        for i, partition_element in enumerate(partition_list):
            sub_dimension = len(partition_element)
            sub_kernel = gpflow.kernels.RBF(
                lengthscales=f64(np.repeat(base_lengthscale, sub_dimension)),
                variance=f64([base_variance]),
                active_dims=partition_element,
            )
            if add_prior:
                a_lengthscale, b_lengthscale = lengthscale_prior_parameters
                a_variance, b_variance = variance_prior_parameters
                sub_kernel.lengthscales.prior = tfd.Gamma(
                    f64(np.repeat(a_lengthscale, sub_dimension)), f64(np.repeat(b_lengthscale, sub_dimension))
                )
                sub_kernel.variance.prior = tfd.Gamma(f64([a_variance]), f64([b_variance]))
            if i == 0:
                self.kernel = sub_kernel
            else:
                self.kernel = self.kernel + sub_kernel

    def __call__(self, X, X2=None, *, full_cov=True, presliced=False):
        return self.kernel(X=X, X2=X2, full_cov=full_cov, presliced=presliced)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return self.kernel(X, X2)

    def K_diag(self, X):
        return self.kernel(X, full_cov=False)


if __name__ == "__main__":
    partition = Partition(5)
    partition.add_partition_element([0, 3])
    partition.add_partition_element([1, 2])
    partition.add_partition_element([4])
    kernel = AdditiveKernel(partition, 1.0, 1.0, True, (1.0, 1.0), (1.0, 1.0))
    print_summary(kernel)
