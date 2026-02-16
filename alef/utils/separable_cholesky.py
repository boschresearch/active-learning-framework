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

import tensorflow as tf
import torch
import gpflow
import gpytorch

from typing import Union


class _SeparableCholeskyDecompositionTF:
    def __init__(self, kernel: gpflow.kernels.Kernel):
        self.kernel = kernel
        self._x = None
        self._L = None

    def _initialize_cholesky(self, x, noise_variance):
        N = tf.shape(x)[-2]
        K = self.kernel(x, full_cov=True)
        noise = noise_variance * tf.eye(N, dtype=K.dtype)

        self._x = x
        self._L = tf.linalg.cholesky(K + noise)

    def _update_cholesky(self, x_new, noise_variance):
        N = tf.shape(x_new)[-2]
        Kst = self.kernel(self._x, x_new)
        noise = noise_variance * tf.eye(N, dtype=Kst.dtype)
        Kt = self.kernel(x_new) + noise

        Ls = self._L
        corner_T = tf.linalg.triangular_solve(Ls, Kst, lower=True, adjoint=False)
        corner = tf.einsum("...ij->...ji", corner_T)
        Lt = tf.linalg.cholesky(Kt - tf.matmul(corner, corner_T))

        self._x = tf.concat([self._x, x_new], axis=-2)
        self._L = tf.concat(
            [tf.concat([Ls, tf.zeros_like(corner_T)], axis=-1), tf.concat([corner, Lt], axis=-1)], axis=-2
        )

    def new_cholesky(self, x, noise_variance):
        if self._x is None:
            self._initialize_cholesky(x, noise_variance)
        else:
            self._update_cholesky(x, noise_variance)
        return self._L


class _SeparableCholeskyDecompositionTorch:
    def __init__(self, kernel: gpytorch.kernels.Kernel, device=torch.device("cpu")):
        self.kernel = kernel
        self.device = device
        self._x = None
        self._L = None

    def _initialize_cholesky(self, x, noise_variance):
        assert isinstance(x, torch.Tensor)
        N = x.size(dim=-2)
        K = self.kernel(x)
        noise = noise_variance * torch.eye(N, device=self.device)

        self._x = x
        self._L = torch.linalg.cholesky(K.evaluate() + noise)

    def _update_cholesky(self, x_new, noise_variance):
        assert isinstance(x_new, torch.Tensor)
        N = x_new.size(dim=-2)
        Kst = self.kernel(self._x, x_new).evaluate()
        noise = noise_variance * torch.eye(N, device=self.device)
        Kt = self.kernel(x_new).evaluate() + noise

        Ls = self._L
        corner_T = torch.linalg.solve_triangular(Ls, Kst, upper=False)
        corner = torch.einsum("...ij->...ji", corner_T)
        Lt = torch.linalg.cholesky(Kt - torch.matmul(corner, corner_T))

        self._x = torch.cat([self._x, x_new], dim=-2)
        self._L = torch.cat(
            (torch.cat((Ls, torch.zeros_like(corner_T)), dim=-1), torch.cat((corner, Lt), dim=-1)), dim=-2
        )

    def new_cholesky(self, x, noise_variance):
        if self._x is None:
            self._initialize_cholesky(x, noise_variance)
        else:
            self._update_cholesky(x, noise_variance)
        return self._L


class SeparableCholeskyDecomposition:
    def __init__(self, kernel: Union[gpflow.kernels.Kernel, gpytorch.kernels.Kernel], device=torch.device("cpu")):
        self.kernel = kernel
        if isinstance(kernel, gpflow.kernels.Kernel):
            self._decomposer = _SeparableCholeskyDecompositionTF(kernel)
        elif isinstance(kernel, gpytorch.kernels.Kernel):
            self._decomposer = _SeparableCholeskyDecompositionTorch(kernel, device)

    def _initialize_cholesky(self, x, noise_variance):
        self._decomposer._initialize_cholesky(x, noise_variance)

    def _update_cholesky(self, x_new, noise_variance):
        self._decomposer._update_cholesky(x_new, noise_variance)

    def new_cholesky(self, x, noise_variance):
        return self._decomposer.new_cholesky(x, noise_variance)
