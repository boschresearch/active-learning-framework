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
import torch.nn as nn
from pyro.distributions import Delta

from alef.configs.base_parameters import INPUT_DOMAIN

"""
The following code is copied from
https://github.com/desi-ivanova/idad/blob/main/neural/modules.py
Copyright (c) 2021 Adam Foster, Desi R. Ivanova, Steven Kleinegesse, licensed under the MIT License,
cf. LICENSE file in the root directory of this source tree).
"""


class LazyDelta(Delta):
    def __init__(self, fn, prototype, log_density=0.0, event_dim=0, validate_args=None):
        self.fn = fn
        super().__init__(
            prototype,
            log_density=log_density,
            event_dim=event_dim,
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LazyDelta, _instance)
        new.fn = self.fn
        batch_shape = torch.Size(batch_shape)
        new.v = self.v.expand(batch_shape + self.event_shape)
        new.log_density = self.log_density.expand(batch_shape)
        super(Delta, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        # The shape of self.v will have expanded along with any .expand calls
        shape = sample_shape + self.v.shape
        output = self.fn()
        return output.expand(shape)

    @property
    def variance(self):
        return torch.zeros_like(self.v)

    def log_prob(self, x):
        return self.log_density


class LazyFn:
    def __init__(self, f, prototype):
        self.f = f
        self.prototype = prototype.clone()

    def lazy(self, *design_obs_pairs):
        def delayed_function():
            return self.f(*design_obs_pairs)

        lazy_delta = LazyDelta(delayed_function, self.prototype, event_dim=self.prototype.dim())
        return lazy_delta


class GeneralizedSigmoid(nn.Module):
    def __init__(
        self,
        input_domain=INPUT_DOMAIN,
    ):
        super().__init__()
        self.a, self.b = input_domain

    def forward(self, x, **kwargs):
        x = nn.Sigmoid()(x)
        return x * (self.b - self.a) + self.a


class GeneralizedTanh(nn.Module):
    def __init__(
        self,
        input_domain=INPUT_DOMAIN,
    ):
        super().__init__()
        self.a, self.b = input_domain

    def forward(self, x, **kwargs):
        x = nn.Tanh()(x)
        return (x + 1) / 2 * (self.b - self.a) + self.a
