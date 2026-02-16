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
from pyro import poutine
from pyro.poutine.util import prune_subsample_sites

from ..base_loss import BaseLoss
from alef.active_learner.amortized_policies.simulated_processes.base_process import BaseSimulatedProcess


"""
The following code is adapted from
https://github.com/desi-ivanova/idad/blob/main/estimators/mi.py
Copyright (c) 2021 Adam Foster, Desi R. Ivanova, Steven Kleinegesse, licensed under the MIT License,
cf. LICENSE file in the root directory of this source tree).
"""


class BaseOEDLoss(BaseLoss):
    def _vectorized(self, fn, *shape, name="vectorization_plate"):
        """
        Wraps a callable inside an outermost :class:`~pyro.plate` to parallelize
        MI computation over `num_particles`, and to broadcast batch shapes of
        sample site functions in accordance with the `~pyro.plate` contexts
        within which they are embedded.
        :param fn: arbitrary callable containing Pyro primitives.
        :return: wrapped callable.
        """

        def wrapped_fn(*args, **kwargs):
            with pyro.plate_stack(name, shape):
                return fn(*args, **kwargs)

        return wrapped_fn

    def set_pyro_model(self, method, name="pyro_model"):
        if self.data_source is None:
            model_v = self._vectorized(method, name=name)
        else:
            data = next(self.data_source)
            if torch.cuda.device_count() > 1:
                assert False, "not implemented"
            model_v = pyro.condition(self._vectorized(method, name=name), data=data)
        return model_v

    def get_rollout_from_pyro_model(self, pyro_model, args, kwargs, graph_type="flat", detach=False):
        trace = poutine.trace(pyro_model, graph_type=graph_type).get_trace(*args, **kwargs)
        if detach:
            trace.detach_()
        trace = prune_subsample_sites(trace)
        return trace

    def get_rollout(
        self, process: BaseSimulatedProcess, args, kwargs, pyro_model_name, graph_type="flat", detach=False
    ):
        model_v = self.set_pyro_model(process.process, name=pyro_model_name)
        trace = self.get_rollout_from_pyro_model(model_v, args, kwargs, graph_type=graph_type, detach=detach)
        return trace

    def get_test_rollout(
        self, process: BaseSimulatedProcess, args, kwargs, pyro_model_name, graph_type="flat", detach=False
    ):
        # this is only used for validation
        model_v = self.set_pyro_model(process.validation, pyro_model_name)
        trace = self.get_rollout_from_pyro_model(model_v, args, kwargs, graph_type=graph_type, detach=detach)
        return trace
