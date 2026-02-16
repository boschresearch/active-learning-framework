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
from torch import nn
from .modules.essential import LazyDelta

"""
The following code is copied from
https://github.com/desi-ivanova/idad/blob/main/neural/baselines.py
Copyright (c) 2021 Adam Foster, Desi R. Ivanova, Steven Kleinegesse, licensed under the MIT License,
cf. LICENSE file in the root directory of this source tree).
"""


########## Baselines #############
# this covers MINEBED and SG-BOED (ACE estimator with prior as proposal)
class DesignBaseline(nn.Module):
    def __init__(self, design_dim):
        super().__init__()
        self.register_buffer("prototype", torch.zeros(design_dim))

    def lazy(self, *design_obs_pairs):
        def delayed_function():
            return self.forward(*design_obs_pairs)

        lazy_delta = LazyDelta(delayed_function, self.prototype, event_dim=self.prototype.dim())
        return lazy_delta


class BatchDesignBaseline(DesignBaseline):
    """
    Batch design baseline: learns T constants.

    - If trained with InfoNCE bound, this is the SG-BOED static baseline.
    - If trained with the NWJ bound, this is the MINEBED static baselines.
    """

    def __init__(
        self,
        T,
        design_dim,
        output_activation=nn.Identity(),
        design_init=torch.distributions.Normal(0, 0.5),
    ):
        super().__init__(design_dim)
        self.designs = nn.ParameterList(
            [nn.Parameter(design_init.sample(torch.zeros(design_dim).shape)) for i in range(T)]
        )
        self.output_activation = output_activation

    def forward(self, *design_obs_pairs):
        j = len(design_obs_pairs)
        return self.output_activation(self.designs[j])


class ConstantBatchBaseline(DesignBaseline):
    def __init__(self, T, design_dim, const_designs_list):
        super().__init__(design_dim)
        self.designs = const_designs_list

    def forward(self, *design_obs_pairs):
        j = len(design_obs_pairs)
        return self.designs[j]


class RandomDesignBaseline(DesignBaseline):
    def __init__(self, T, design_dim, random_designs_dist=None):
        super().__init__(design_dim)
        if random_designs_dist is None:
            random_designs_dist = torch.distributions.Normal(torch.zeros(design_dim), torch.ones(design_dim))
        self.random_designs_dist = random_designs_dist

    def forward(self, *design_obs_pairs):
        return self.random_designs_dist.sample()
