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

from copy import deepcopy
from typing import Tuple
import gpytorch
import torch
import numpy as np
from alef.kernels.pytorch_kernels.elementary_kernels_pytorch import BaseElementaryKernelPytorch, RBFKernelPytorch
from gpytorch.constraints import Positive


class HierarchicalHyperplaneKernelPytorch(gpytorch.kernels.Kernel):
    def __init__(
        self,
        base_kernel: BaseElementaryKernelPytorch,
        input_dimension: int,
        base_hyperplane_mu: float,
        base_hyperplane_std: float,
        base_smoothing: float,
        topology: int,
        add_prior: bool,
        smoothing_prior_parameters: Tuple[float, float],
        **kwargs,
    ):
        super().__init__()
        self.input_dimension = input_dimension
        self.smoothing_prior_parameters = smoothing_prior_parameters
        self.base_hyperplane_mu = base_hyperplane_mu
        self.base_hyperplane_std = base_hyperplane_std
        self.base_smoothing = base_smoothing
        self.add_prior = add_prior
        self.topology = topology
        self.set_default_topology(topology)
        self.kernel_list = []
        for i in range(0, self.n_experts):
            kernel = deepcopy(base_kernel)
            self.kernel_list.append(kernel)
        self.kernel_list = torch.nn.ModuleList(self.kernel_list)
        self.initialize_hyperplane_parameters()

    def initialize_hyperplane_parameters(self):
        self.register_hyperplane_params()
        self.register_smoothing_params()

    def register_smoothing_params(self):
        smoothing_constraint = Positive()
        self.register_parameter(name="raw_smoothing", parameter=torch.nn.Parameter(torch.ones(self.M)))
        self.register_constraint("raw_smoothing", smoothing_constraint)
        if self.add_prior:
            self.register_prior(
                "smoothing_prior",
                gpytorch.priors.GammaPrior(self.smoothing_prior_parameters[0], self.smoothing_prior_parameters[1]),
                lambda m: m.smoothing,
                lambda m, v: m._set_smoothing(v),
            )

    def register_hyperplane_params(self):
        self.register_parameter(
            name="hyperplanes", parameter=torch.nn.Parameter(torch.randn(self.input_dimension + 1, self.M) * 0.1)
        )
        if self.add_prior:
            self.register_prior(
                "hyperplanes_prior",
                gpytorch.priors.NormalPrior(0.0, 1.0),
                lambda m: m.hyperplanes,
                lambda m, v: m._set_hyperplanes(v),
            )

    def _set_hyperplanes(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.hyperplanes)
        self.initialize(hyperplanes=value)

    @property
    def smoothing(self):
        return self.raw_smoothing_constraint.transform(self.raw_smoothing)

    @smoothing.setter
    def smoothing(self, value):
        return self._set_smoothing(value)

    def _set_smoothing(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_smoothing)
        self.initialize(raw_smoothing=self.raw_smoothing_constraint.inverse_transform(value))

    def set_default_topology(self, n_depth):
        if n_depth == 1:
            left_matrix = np.array([[1], [0]])
            right_matrix = np.array([[0], [1]])
            self.set_topology_matrices(left_matrix, right_matrix)
        elif n_depth == 2:
            left_matrix = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]])
            right_matrix = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1]])
            self.set_topology_matrices(left_matrix, right_matrix)
        elif n_depth == 3:
            left_matrix = np.array(
                [
                    [1, 1, 0, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            )
            right_matrix = np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0],
                    [1, 0, 1, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 1],
                ]
            )
            self.set_topology_matrices(left_matrix, right_matrix)

    def set_topology_matrices(self, left_matrix, right_matrix):
        self.left_matrix = left_matrix
        self.right_matrix = right_matrix
        self.n_experts = self.left_matrix.shape[0]  ## Number of experts
        self.M = self.left_matrix.shape[1]  ## Number of gate nodes
        assert self.n_experts == self.right_matrix.shape[0]
        assert (self.left_matrix.shape[0] * self.left_matrix.shape[1]) == (
            (self.left_matrix == 0).sum() + (self.left_matrix == 1).sum()
        )
        assert (self.right_matrix.shape[0] * self.right_matrix.shape[1]) == (
            (self.right_matrix == 0).sum() + (self.right_matrix == 1).sum()
        )
        self.left_matrix = torch.from_numpy(self.left_matrix)
        self.right_matrix = torch.from_numpy(self.right_matrix)

    def sigmoid(self, x, smoothing):
        return 1 / (1 + torch.exp(-1 * x * smoothing))

    def gate(self, X: torch.Tensor):
        # X dim: N x D
        N = X.shape[0]
        expert_probabilities = []
        for k in range(0, self.n_experts):
            prob = torch.ones(N)  # N
            for j in range(0, self.M):
                w = self.hyperplanes[:, j]
                w_0 = w[0]
                w_rest = w[1:].unsqueeze(0)
                smoothing = self.smoothing[j]
                prob_elem = torch.pow(
                    self.sigmoid(w_0 + torch.matmul(X, torch.transpose(w_rest, 0, 1)), smoothing),
                    self.left_matrix[k, j],
                ) * torch.pow(
                    1 - self.sigmoid(w_0 + torch.matmul(X, torch.transpose(w_rest, 0, 1)), smoothing),
                    self.right_matrix[k, j],
                )
                prob = prob * torch.squeeze(prob_elem)
            prob = prob.unsqueeze(-1)
            expert_probabilities.append(prob)
        return expert_probabilities

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert not last_dim_is_batch
        gate_x = self.gate(x1)
        if diag:
            output = torch.zeros((x1.shape[0],))
            for k in range(0, self.n_experts):
                kernel_diag = self.kernel_list[k].forward(x1, x2, diag=True)
                output += torch.multiply(torch.squeeze(torch.pow(gate_x[k], 2.0)), kernel_diag)
                return output
        gate_x2 = self.gate(x2)
        output = torch.zeros((x1.shape[0], x2.shape[0]))
        for k in range(0, self.n_experts):
            output += torch.matmul(gate_x[k], torch.transpose(gate_x2[k], 0, 1)) * self.kernel_list[k].forward(
                x1, x2, diag=False
            )
        return output


if __name__ == "__main__":
    rbf_kernel_pytorch = RBFKernelPytorch(3, 1.0, 1.0, True, (2.0, 2.0), (2.0, 2.0), False, 0, "RBF")
    hhk = HierarchicalHyperplaneKernelPytorch(rbf_kernel_pytorch, 3, 0.0, 1.0, 1.0, 2, True, (2.0, 2.0))
    X = torch.randn((20, 3))
    X2 = torch.randn((10, 3))
    print(hhk.forward(X, X, True).detach().numpy())
