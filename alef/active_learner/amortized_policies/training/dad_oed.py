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
import pyro
from pyro.infer.util import torch_item

"""
The following code is copied from
https://github.com/ae-foster/dad/blob/main/oed/design.py
Copyright (c) 2021 Adam Foster and Desi R. Ivanova, licensed under the MIT License,
cf. LICENSE file in the root directory of this source tree).
"""


class OED:
    def __init__(self, process, optim, loss, **kwargs):
        self.process = process
        self.optim = optim
        self.loss = loss
        super().__init__(**kwargs)

        if not isinstance(optim, pyro.optim.PyroOptim):
            raise ValueError("Optimizer should be an instance of pyro.optim.PyroOptim class.")

    def evaluate_loss(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float
        Evaluate the loss function. Any args or kwargs are passed to the process and guide.
        """
        with torch.no_grad():
            loss = self.loss.loss(self.process, *args, **kwargs)
            if isinstance(loss, tuple):
                # Support losses that return a tuple
                return type(loss)(map(torch_item, loss))
            else:
                return torch_item(loss)

    def step(self, clip_grads=True, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float
        Take a gradient step on the loss function (and any auxiliary loss functions
        generated under the hood by `loss_and_grads`).
        Any args or kwargs are passed to the process and guide
        """

        with pyro.poutine.trace(param_only=True) as param_capture:
            loss = self.loss.differentiable_loss(self.process, *args, **kwargs)
            loss.backward()

        params = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values())
        # gradient clipping: The norm is computed over all gradients together,
        # as if they were concatenated into a single vector.
        # if clip_grads:
        #    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0, norm_type="inf")

        # actually perform gradient steps
        # torch.optim objects gets instantiated for any params that haven't been seen yet
        self.optim(params)

        # zero gradients
        pyro.infer.util.zero_grads(params)

        if isinstance(loss, tuple):
            # Support losses that return a tuple, e.g. ReweightedWakeSleep.
            return type(loss)(map(torch_item, loss))
        else:
            return torch_item(loss)

    def validation(self, *args, **kwargs):
        return self.loss.validation(self.process, *args, **kwargs)
