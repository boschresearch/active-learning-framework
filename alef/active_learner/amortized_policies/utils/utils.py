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

import torch
from torch.nn import Module, DataParallel
from linear_operator.operators import CatLinearOperator, DiagLinearOperator


def wrap_parallel(module: Module, iscuda: bool, dim: int):
    if iscuda:
        return DataParallel(module, dim=dim).to(torch.device("cuda"))
    else:
        return module


def compute_kernel_batch(
    kernel_list,
    x1,
    x2,
    noise_var_list=None,
    *,
    batch_dim: int = 1,
    gram_size_dim: int = -2,
    return_linear_operator: bool = False,
):
    assert x1.device == x2.device
    device = x1.device

    x1_seq = x1.unbind(batch_dim)
    x2_seq = x2.unbind(batch_dim)
    assert len(kernel_list) == len(x1_seq) == len(x2_seq)

    if not noise_var_list is None:
        assert len(kernel_list) == len(noise_var_list)
        assert x1.shape[gram_size_dim] == x2.shape[gram_size_dim]
        K = []
        for i, kernel in enumerate(kernel_list):
            Ki = kernel(x1_seq[i], x2_seq[i]).unsqueeze(batch_dim)
            vi = DiagLinearOperator(noise_var_list[i] * torch.ones(x1.shape[gram_size_dim], device=x1.device))
            K.append(Ki + vi)
    else:
        K = []
        for i, kernel in enumerate(kernel_list):
            Ki = kernel(x1_seq[i], x2_seq[i]).unsqueeze(batch_dim)
            K.append(Ki)

    out = CatLinearOperator(*K, dim=batch_dim, output_device=device)
    if return_linear_operator:
        return out
    else:
        return out.to_dense()
