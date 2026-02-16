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
import torch.nn as nn

"""
The following code is adapted from
https://github.com/desi-ivanova/idad/blob/main/neural/modules.py
Copyright (c) 2021 Adam Foster, Desi R. Ivanova, Steven Kleinegesse, licensed under the MIT License,
cf. LICENSE file in the root directory of this source tree).
"""


class MLPEmitter(nn.Module):
    def __init__(
        self,
        input_dim,  # embedding dim so E
        hidden_dim,
        output_dim,  # [1, D]
        activation=nn.ReLU(),
        output_activation=nn.Identity(),
        batch_norm=False,
        name=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.input_dim_flat = input_dim
        self.output_dim = output_dim
        self.output_dim_flat = np.prod(output_dim)
        self._name = name
        self.batch_norm = batch_norm

        ### Layers
        ### Layers
        if isinstance(hidden_dim, int):
            self.linear1 = nn.Linear(self.input_dim_flat, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim) if self.batch_norm else nn.Identity()
        else:
            self.linear1 = nn.Linear(self.input_dim_flat, hidden_dim[0])
            self.bn1 = nn.BatchNorm1d(hidden_dim[0]) if self.batch_norm else nn.Identity()
        self.activation = activation

        if isinstance(hidden_dim, int):
            self.middle = nn.Identity()
            self.output_layer = nn.Linear(hidden_dim, self.output_dim_flat)
        else:
            self.middle = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Linear(hidden_dim[i], hidden_dim[i + 1]),
                        nn.BatchNorm1d(hidden_dim[i + 1]) if self.batch_norm else nn.Identity(),
                        self.activation,
                    )
                    for i in range(0, len(hidden_dim) - 1)
                ]
            )
            self.output_layer = nn.Linear(hidden_dim[-1], self.output_dim_flat)

        self.output_activation = output_activation
        self.hidden_dim = hidden_dim

    def forward(self, x, **kwargs):
        inputs = x
        x = self.linear1(inputs)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.middle(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x.reshape(x.shape[:-1] + self.output_dim)
